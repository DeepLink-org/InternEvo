#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from internlm.accelerator import get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.ops.fusion_ops_import_helper import try_import_fused_rotary

from ..utils import gather_forward_split_backward, split_forward_gather_backward

internlm_accelerator = get_accelerator()

apply_rotary_emb, apply_rotary_emb_qkv_, apply_rotary_func = None, None, None


class Embedding1D(nn.Module):
    """
    1D Embedding.

    Args:
        num_embeddings (int): The size of vocab.
        embedding_dim (int): The dimention of model.
        padding_idx (int): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                            therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                            i.e. it remains as a fixed "pad". None by default.
        dtype (Optional[torch.dtype]): Data type None by default.
        embed_split_hidden (Optional[Bool]): Whether to split the embed_dim in tensor parallel style.

    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *args,
        padding_idx: int = None,
        dtype: torch.dtype = None,
        embed_split_hidden: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        self.embed_split_hidden = embed_split_hidden
        if self.embed_split_hidden:
            self.embed_split_hidden = gpc.tensor_parallel_size > 1

        split_nums = 1 if not self.embed_split_hidden else gpc.tensor_parallel_size
        embed_dim_per_partition = embedding_dim // split_nums

        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs

        self.weight = nn.Parameter(torch.empty((num_embeddings, embed_dim_per_partition), dtype=dtype))

    def forward(self, input_: Tensor) -> Tensor:
        output = F.embedding(input_, self.weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)

        if self.embed_split_hidden:
            output = gather_forward_split_backward(output, ParallelMode.TENSOR, dim=-1)

        if gpc.config.parallel.sequence_parallel:
            output = split_forward_gather_backward(output, ParallelMode.TENSOR, dim=1)

        return output


def _torch_apply_rotary_func(
    x1: torch.Tensor,
    x2: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    conj: bool = False,
):
    assert x1.device == x2.device == cos.device == sin.device, "All inputs must be on the same device"
    assert x1.dtype == x2.dtype == cos.dtype == sin.dtype, "All inputs must have the same dtype"
    assert x1.size() == x2.size(), "Input x1 and x2 must have the same sizes"
    assert cos.size() == sin.size(), "Input cos and sin must have the same sizes"

    x1, x2, cos, sin = x1.float(), x2.float(), cos.float(), sin.float()

    if conj:
        out1.copy_(x1 * cos + x2 * sin)
        out2.copy_(-x1 * sin + x2 * cos)
    else:
        out1.copy_(x1 * cos - x2 * sin)
        out2.copy_(x1 * sin + x2 * cos)

    return out1, out2


class ApplyRotaryEmb(torch.autograd.Function):
    """
    ApplyRotaryEmb
    """

    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False):
        """
            x: (batch_size, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        _, seqlen, _, headdim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)
        x_ro = x[..., :rotary_dim]
        x1, x2 = x_ro.chunk(2, dim=-1) if not interleaved else (x_ro[..., ::2], x_ro[..., 1::2])
        out = torch.empty_like(x)
        out_ro = out[..., :rotary_dim]
        o1, o2 = out_ro.chunk(2, dim=-1) if not interleaved else (out_ro[..., ::2], out_ro[..., 1::2])

        apply_rotary_func(
            x1,
            x2,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            o1,
            o2,
            False,
        )

        if rotary_dim < headdim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        return out

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        _, seqlen, _, headdim = do.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        do_ro = do[..., :rotary_dim]
        do1, do2 = do_ro.chunk(2, dim=-1) if not ctx.interleaved else (do_ro[..., ::2], do_ro[..., 1::2])
        dx = torch.empty_like(do)
        dx_ro = dx[..., :rotary_dim]
        dx1, dx2 = dx_ro.chunk(2, dim=-1) if not ctx.interleaved else (dx_ro[..., ::2], dx_ro[..., 1::2])

        apply_rotary_func(
            do1,
            do2,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            dx1,
            dx2,
            True,
        )
        if rotary_dim < headdim:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
        return dx, None, None, None, None


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    """
    ApplyRotaryEmbQKV_
    """

    @staticmethod
    def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None, interleaved=False):
        """
            qkv: (total, 3, nheads, headdim) / (batch_size, seqlen, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
                1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """
        # len(qkv.shape) == 4 means the format of qkv is (total, 3, nheads, headdim) which is packed,
        # otherwise the format of qkv is (batch_size, seqlen, 3, nheads, headdim) which is unpacked.
        # We handle both packed qkv and unpacked qkv scenario in this class.
        three = qkv.shape[1] if len(qkv.shape) == 4 else qkv.shape[2]
        assert three == 3
        seqlen = None if len(qkv.shape) == 4 else qkv.shape[1]
        rotary_seqlen, rotary_dim = cos.shape
        if len(qkv.shape) != 4:
            assert seqlen <= rotary_seqlen
        headdim = qkv.shape[-1]
        rotary_dim *= 2
        assert rotary_dim <= headdim
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)
        q_ro = qkv[:, 0, :, :rotary_dim] if len(qkv.shape) == 4 else qkv[:, :, 0, :, :rotary_dim]
        q1, q2 = q_ro.chunk(2, dim=-1) if not interleaved else (q_ro[..., ::2], q_ro[..., 1::2])
        re_cos = rearrange(cos, "s d -> s 1 d") if len(qkv.shape) == 4 else rearrange(cos[:seqlen], "s d -> s 1 d")
        re_sin = rearrange(sin, "s d -> s 1 d") if len(qkv.shape) == 4 else rearrange(sin[:seqlen], "s d -> s 1 d")

        apply_rotary_func(q1, q2, re_cos, re_sin, q1, q2, False)

        k_ro = qkv[:, 1, :, :rotary_dim] if len(qkv.shape) == 4 else qkv[:, :, 1, :, :rotary_dim]
        k1, k2 = k_ro.chunk(2, dim=-1) if not interleaved else (k_ro[..., ::2], k_ro[..., 1::2])
        re_cos_k = (
            rearrange(cos_k, "s d -> s 1 d") if len(qkv.shape) == 4 else rearrange(cos_k[:seqlen], "s d -> s 1 d")
        )
        re_sin_k = (
            rearrange(sin_k, "s d -> s 1 d") if len(qkv.shape) == 4 else rearrange(sin_k[:seqlen], "s d -> s 1 d")
        )

        apply_rotary_func(k1, k2, re_cos_k, re_sin_k, k1, k2, False)

        ctx.save_for_backward(cos, sin, cos_k, sin_k)
        ctx.interleaved = interleaved
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        cos, sin, cos_k, sin_k = ctx.saved_tensors
        seqlen = None if len(dqkv.shape) == 4 else dqkv.shape[1]
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        dq_ro = dqkv[:, 0, :, :rotary_dim] if len(dqkv.shape) == 4 else dqkv[:, :, 0, :, :rotary_dim]
        dq1, dq2 = dq_ro.chunk(2, dim=-1) if not ctx.interleaved else (dq_ro[..., ::2], dq_ro[..., 1::2])
        re_cos = rearrange(cos, "s d -> s 1 d") if len(dqkv.shape) == 4 else rearrange(cos[:seqlen], "s d -> s 1 d")
        re_sin = rearrange(sin, "s d -> s 1 d") if len(dqkv.shape) == 4 else rearrange(sin[:seqlen], "s d -> s 1 d")

        apply_rotary_func(dq1, dq2, re_cos, re_sin, dq1, dq2, True)

        dk_ro = dqkv[:, 1, :, :rotary_dim] if len(dqkv.shape) == 4 else dqkv[:, :, 1, :, :rotary_dim]
        dk1, dk2 = dk_ro.chunk(2, dim=-1) if not ctx.interleaved else (dk_ro[..., ::2], dk_ro[..., 1::2])
        re_cos_k = (
            rearrange(cos_k, "s d -> s 1 d") if len(dqkv.shape) == 4 else rearrange(cos_k[:seqlen], "s d -> s 1 d")
        )
        re_sin_k = (
            rearrange(sin_k, "s d -> s 1 d") if len(dqkv.shape) == 4 else rearrange(sin_k[:seqlen], "s d -> s 1 d")
        )

        apply_rotary_func(dk1, dk2, re_cos_k, re_sin_k, dk1, dk2, True)

        return dqkv, None, None, None, None, None


apply_rotary_emb, apply_rotary_emb_qkv_, apply_rotary_func = try_import_fused_rotary()
if apply_rotary_emb is None:
    apply_rotary_emb = ApplyRotaryEmb.apply
if apply_rotary_emb_qkv_ is None:
    apply_rotary_emb_qkv_ = ApplyRotaryEmbQKV_.apply
if apply_rotary_func is None:
    apply_rotary_func = _torch_apply_rotary_func


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base > 0, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(self, dim: int, base=10000, scale_base=0, device=None):
        """ """
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.scale_base = scale_base
        self.scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base > 0
            else None
        )

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _update_cos_sin_cache(self, x, indexes):
        """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim)"""
        if not isinstance(indexes, int):
            seqlen = indexes.max().item() + 1
        else:
            seqlen = indexes + 1  # eval_forward
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seqlen > self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=x.device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(x.dtype)
                self._sin_cached = torch.sin(freqs).to(x.dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

    def forward(self, qkv: torch.Tensor, **kwargs):
        if kwargs.get("indexes", None) is not None:
            return self._forward(qkv, kwargs.pop("indexes"))
        if kwargs.get("inference_params", None) is not None:
            return self._eval_forward(qkv, seqlen_offset=kwargs.get("inference_params", None).sequence_len_offset)
        else:
            return self._eval_forward(qkv)

    def _forward(self, qkv: torch.Tensor, indexes=0) -> Tuple[torch.Tensor, torch.Tensor]:
        self._update_cos_sin_cache(qkv, indexes)
        if self.scale is None:
            return apply_rotary_emb_qkv_(qkv, self._cos_cached[indexes], self._sin_cached[indexes])
        else:
            return apply_rotary_emb_qkv_(
                qkv,
                self._cos_cached[indexes],
                self._sin_cached[indexes],
                self._cos_k_cached[indexes],
                self._sin_k_cached[indexes],
            )

    def _eval_forward(self, qkv, seqlen_offset=0):
        """
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        self._update_cos_sin_cache(qkv, seqlen_offset + qkv.shape[1])
        if self.scale is None:
            return apply_rotary_emb_qkv_(qkv, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:])
        else:
            return apply_rotary_emb_qkv_(
                qkv,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
                self._cos_k_cached[seqlen_offset:],
                self._sin_k_cached[seqlen_offset:],
            )

    def _single_forward(self, x, indexes=0):
        assert self.scale is None
        self._update_cos_sin_cache(x, indexes)
        ret = apply_rotary_emb(x, self._cos_cached[indexes], self._sin_cached[indexes])
        return ret

    def _single_eval_forward(self, x, seqlen_offset=0):
        assert self.scale is None
        self._update_cos_sin_cache(x, seqlen_offset + x.shape[1])
        return apply_rotary_emb(x, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:])


class LinearRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev.

    Reference implementation:
        https://github.com/huggingface/transformers/blob/200009566639b5a83604e522a41df3a9 \
            5b6056ed/src/transformers/models/llama/modeling_llama.py#L159C1-L176C1
    """

    def __init__(
        self, dim: int, base=10000, scale_base=0, device=None, max_position_embeddings=2048, scaling_factor=1.0
    ):
        super().__init__(dim=dim, base=base, scale_base=scale_base, device=device)
        self.max_position_embeddings = max_position_embeddings
        self.scaling_factor = scaling_factor

    def _update_cos_sin_cache(self, x, indexes):
        """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim)"""
        if not isinstance(indexes, int):
            seqlen = indexes.max().item() + 1
        else:
            seqlen = indexes + 1

        t = torch.arange(seqlen, device=x.device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq.to(device=t.device))
        if self.scale is None:
            self._cos_cached = torch.cos(freqs).to(x.dtype)
            self._sin_cached = torch.sin(freqs).to(x.dtype)
        else:
            power = (
                torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
            ) / self.scale_base
            scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
            # We want the multiplication by scale to happen in fp32
            self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
            self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
            self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
            self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla.

    Reference implementation:
        https://github.com/huggingface/transformers/blob/eb8489971ac1415f67b0abdd1584fde8 \
            b659ced9/src/transformers/models/llama/modeling_llama.py#L147
    """

    def __init__(
        self, dim: int, base=10000, scale_base=0, device=None, max_position_embeddings=2048, scaling_factor=1.0
    ):
        super().__init__(dim=dim, base=base, scale_base=scale_base, device=device)
        self.max_position_embeddings = max_position_embeddings
        self.scaling_factor = scaling_factor

    def _update(self, seqlen, x):
        self._seq_len_cached = seqlen
        if seqlen > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seqlen / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(x.device) / self.dim))
        else:
            inv_freq = self.inv_freq

        t = torch.arange(seqlen, device=x.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq.to(device=t.device))
        if self.scale is None:
            self._cos_cached = torch.cos(freqs).to(x.dtype)
            self._sin_cached = torch.sin(freqs).to(x.dtype)
        else:
            power = (
                torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
            ) / self.scale_base
            scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
            # We want the multiplication by scale to happen in fp32
            self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
            self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
            self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
            self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

    def _update_cos_sin_cache(self, x, indexes):
        """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim)"""
        if not isinstance(indexes, int):
            seqlen = indexes.max().item() + 1
        else:
            seqlen = indexes + 1  # eval_forward
        if seqlen <= self.max_position_embeddings:
            # Reset the tables if the sequence length has changed,
            # or if we're on a new device (possibly due to tracing for instance)
            if (
                self._seq_len_cached > self.max_position_embeddings
                or seqlen > self._seq_len_cached
                or self._cos_cached.device != x.device
                or self._cos_cached.dtype != x.dtype
            ):
                self._update(seqlen, x)
        else:
            self._update(seqlen, x)
