from .abstract_accelerator import Accelerator, AcceleratorType

try:
    import torch.ptpu
except ImportError:
    pass


class PTPU_Accelerator(Accelerator):
    """Accelerator for PTPU device.

    Args:
        Accelerator (Accelerator): _description_
    """

    def __init__(self) -> None:
        self._name_str = "ptpu"
        self._communication_backend_name = "pccl"
        self.amp = self.get_amp()
        # self.memory = torch.ptpu.memory
        # self._find_or_mock_module("flash_attn_2_cuda")

    def _find_or_mock_module(self, module_name) -> bool:
        import importlib.util
        import sys
        import types

        """Find or mock a module. Return True if the module is found, False otherwise."""
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None:
            sys.modules[module_name] = types.SimpleNamespace()  # type: ignore
        return module_spec is not None

    def get_backend_name(self):
        """
        Return the name of the accelerator.
        """
        return self._name_str

    def get_accelerator_backend(self):
        """
        Return the name of the backend.
        """
        return AcceleratorType.PTPU

    # Device APIs
    def device_name(self, device_index=None):
        """
        Return the name of the device.
        """
        if device_index is None:
            return "ptpu"
        return "ptpu:{}".format(device_index)

    def set_device(self, device_index):
        """
        Bind the current process to a device.
        """
        torch.ptpu.set_device(device_index)

    def get_device_id(self):
        """
        Return the current device index.
        """
        return torch.ptpu.current_device()

    def current_device_name(self):
        """
        Return the name of the current device.
        """
        return "ptpu:{}".format(torch.ptpu.current_device())

    def device_count(self):
        """
        Return the number of devices on the machine.
        """
        return torch.ptpu.device_count()

    def synchronize(self, device_index=None):
        """
        Synchronize the current process.
        """
        return torch.ptpu.synchronize(device_index)

    # RNG APIs
    def random(self):
        """
        Get random number.
        """
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        """
        Sets the random number generator state of the specified GPU.
        """
        if device_index is None:
            return torch.ptpu.set_rng_state(new_state)

        return torch.ptpu.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        """
        Returns the random number generator state of the specified GPU as a ByteTensor.
        """
        if device_index is None:
            return torch.ptpu.get_rng_state()

        return torch.ptpu.get_rng_state(device_index)

    def manual_seed(self, seed):
        """
        Sets the seed for generating random numbers for the current GPU.
        """
        return torch.manual_seed(seed)

    def manual_seed_all(self, seed):
        """
        Set the random seed for the all processes.
        """
        return torch.ptpu.manual_seed_all(seed)

    def initial_seed(self):
        """
        Returns the current random seed of the current GPU.
        """
        return torch.ptpu.initial_seed()

    def default_generator(self, device_index):
        """
        Returns the default generators according to device index
        """
        return torch.ptpu.default_generators[device_index]

    # Streams/Events
    @property
    def Stream(self):
        """
        A CUDA stream is a linear sequence of execution that belongs to
        a specific device, independent from other streams.
        """
        return torch.ptpu.Stream

    def stream(self, _stream):
        """
        Wrapper around the Context-manager StreamContext that selects a given stream.
        """
        return torch.ptpu.stream(_stream)

    def current_stream(self, device_index=None):
        """
        Returns the currently selected Stream for a given device.
        """
        return torch.ptpu.current_stream(device_index)

    def default_stream(self, device_index=None):
        """
        Returns the default Stream for a given device.
        """
        return torch.ptpu.default_stream(device_index)

    @property
    def Event(self):
        """
        CUDA events are synchronization markers that can be used
        to monitor the device's progress, to accurately measure timing,
        and to synchronize CUDA streams.
        """
        return torch.ptpu.Event

    # Memory management
    def empty_cache(self):
        """
        Releases all unoccupied cached memory currently held by the caching allocator
        so that those can be used in other GPU application and visible in nvidia-smi.
        """
        return torch.ptpu.empty_cache()

    def memory_allocated(self, device_index=None):
        """
        Returns the current GPU memory occupied by tensors in bytes for a given device.
        """
        return torch.ptpu.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        """
        Returns the maximum GPU memory occupied by tensors in bytes for a given device.
        """
        return torch.ptpu.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        """
        Resets the starting point in tracking maximum GPU memory occupied by
        tensors for a given device.
        """
        return torch.ptpu.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        """
        Returns the cached memory
        """
        return torch.ptpu.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):
        """
        Returns the maximum cached memory
        """
        return torch.ptpu.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):
        """
        Resets the starting point in tracking maximum GPU memory managed by
        the caching allocator for a given device.
        """
        return torch.ptpu.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):
        """
        Returns the memory stats
        """
        if hasattr(torch.ptpu, "memory_stats"):
            return torch.ptpu.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        """
        Resets the "peak" stats tracked by the CUDA memory allocator.
        """
        if hasattr(torch.ptpu, "reset_peak_memory_stats"):
            return torch.ptpu.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        """
        Returns the current GPU memory managed by the caching allocator
        in bytes for a given device.
        """
        if hasattr(torch.ptpu, "memory_reserved"):
            return torch.ptpu.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        """
        Returns the maximum GPU memory managed by the caching allocator
        in bytes for a given device.
        """
        if hasattr(torch.ptpu, "max_memory_reserved"):
            return torch.ptpu.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        """
        Returns the total memory
        """
        return torch.ptpu.get_device_properties(device_index).total_memory

    # Data types
    def is_bf16_supported(self):
        """
        Returns true if bf16 is supported. Otherwise, returns false
        """
        return torch.ptpu.is_bf16_supported()

    def is_fp16_supported(self):
        """
        Returns true if fp16 is supported. Otherwise, returns false
        """
        major, _ = torch.ptpu.get_device_capability()
        return bool(major >= 7)

    # Misc
    def get_amp(self):
        """
        Returns the 'amp' module from torch.cuda if available, else returns None.
        """
        if hasattr(torch.cuda, "amp"):
            return torch.cuda.amp
        return None

    def is_available(self):
        """
        Checks and returns True if CUDA is available, False otherwise.
        """
        return torch.ptpu.is_available()

    def range_push(self, msg):
        """
        If available, pushes a range with the given message for profiling using NVTX.
        """
        if hasattr(torch.ptpu.nvtx, "range_push"):
            return torch.ptpu.nvtx.range_push(msg)

    def range_pop(self):
        """
        If available, pops the most recent range pushed using NVTX.
        """
        if hasattr(torch.ptpu.nvtx, "range_pop"):
            return torch.ptpu.nvtx.range_pop()

    def lazy_call(self, callback):
        """
        Executes the given callback with lazy propagation if available.
        """
        return torch.ptpu._lazy_call(callback)

    def communication_backend_name(self):
        """
        Returns the name of the current communication backend.
        """
        return self._communication_backend_name

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        """
        Property to get the BFloat16Tensor class from torch.cuda.
        """
        return torch.ptpu.BFloat16Tensor

    @property
    def ByteTensor(self):
        """
        Property to get the ByteTensor class from torch.cuda.
        """
        return torch.ptpu.ByteTensor

    @property
    def DoubleTensor(self):
        """
        Property to get the DoubleTensor class from torch.cuda.
        """
        return torch.ptpu.DoubleTensor

    @property
    def FloatTensor(self):
        """
        Property to get the FloatTensor class from torch.cuda.
        """
        return torch.ptpu.FloatTensor

    @property
    def HalfTensor(self):
        """
        Property to get the HalfTensor class from torch.cuda.
        """
        return torch.ptpu.HalfTensor

    @property
    def IntTensor(self):
        """
        Property to get the IntTensor class from torch.cuda.
        """
        return torch.ptpu.IntTensor

    @property
    def LongTensor(self):
        """
        Property to get the LongTensor class from torch.cuda.
        """
        return torch.ptpu.LongTensor

    def pin_memory(self, tensor):
        """
        Pins the memory of the given tensor, if it's a CUDA tensor.
        """
        return tensor.pin_memory()

    def on_accelerator(self, tensor):
        """
        Checks and returns True if the given tensor is on an accelerator (CUDA device), False otherwise.
        """
        device_str = str(tensor.device)
        return bool(device_str.startswith("ptpu:"))

    def set_allow_tf32(self, enable: bool):
        """
        Sets the `allow_tf32` flag in cuDNN and CUDA matrix multiplication to the given boolean value.
        """
        print(f"Not support tf32 for PTPU, {enable}!")

    def return_custom_bwd(self):
        """
        Returns the custom backward hook function from torch.cuda.amp, if available.
        """
        return torch.cuda.amp.custom_bwd

    def return_custom_fwd(self):
        """
        Returns the custom forward hook function from torch.cuda.amp, if available.
        """
        return torch.cuda.amp.custom_fwd
