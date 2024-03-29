JOB_NAME = "70b_train"
DO_ALERT = False
SEQ_LEN = 4096


model_type = "INTERNLM"
model_type = "LLAMA2"
if model_type == "LLAMA2":
    HIDDEN_SIZE = 8192
    NUM_ATTENTION_HEAD = 64
    NUM_KV_ATTENTION_HEAD = 8
    MLP_RATIO = 3.5
    # NUM_LAYER = 80
    NUM_LAYER = 8
    VOCAB_SIZE = 32000
elif model_type == "INTERNLM2_PUBLIC":
    #70B
    # HIDDEN_SIZE = 8192
    # NUM_ATTENTION_HEAD = 64
    # NUM_KV_ATTENTION_HEAD = 8
    # MLP_RATIO = 3.5
    # NUM_LAYER = 80
    # VOCAB_SIZE = 92544

    #100B
    VOCAB_SIZE = 92544
    HIDDEN_SIZE = 8192
    NUM_ATTENTION_HEAD = 64
    NUM_KV_ATTENTION_HEAD = 8
    MLP_RATIO = 4.5
    NUM_LAYER = 96
else:
    HIDDEN_SIZE = 8192
    NUM_ATTENTION_HEAD = 64
    MLP_RATIO = 8 / 3
    NUM_LAYER = 80
    VOCAB_SIZE = 103168

MODEL_ONLY_FOLDER = "local:llm_ckpts_ljx_16dp_bf16_2/3500"
SAVE_CKPT_FOLDER = "local:llm_ckpts_ljx_16dp_bf16_2"
LOAD_CKPT_FOLDER = "local:llm_ckpts_ljx_16dp_bf16_2/3500"

CHECKPOINT_EVERY = 500
ckpt = dict(
    enable_save_ckpt=False,  # enable ckpt save.
    save_ckpt_folder=SAVE_CKPT_FOLDER,  # Path to save training ckpt.
    auto_resume=False,
    checkpoint_every=CHECKPOINT_EVERY,
    async_upload=True,  # async ckpt upload. (only work for boto3 ckpt)
    async_upload_tmp_folder="/dev/shm/internlm_tmp_ckpt/",  # path for temporarily files during asynchronous upload.
    oss_snapshot_freq=int(CHECKPOINT_EVERY / 2),  # snapshot ckpt save frequency.
)

# TRAIN_FOLDER = "/data/wangguoteng/opensource_data/0711_scratch_tokenized_refineweb_small/train/"  # "/path/to/dataset"
# TRAIN_FOLDER = "/data/wangguoteng/opensource_data/0711_scratch_tokenized_refineweb_small/train/"  # "/path/to/dataset"
# TRAIN_FOLDER = "/data/wangguoteng/InternEvo/mnt/petrelfs/share_data/wangguoteng.p/new_wiki_dataset_v7/train"
# TRAIN_FOLDER = "/data/lijiaxing/alpaca_token/train"
# TRAIN_FOLDER = None
data = dict(
    seq_len=SEQ_LEN,
    # micro_num means the number of micro_batch contained in one gradient update
    micro_num=8,
    # packed_length = micro_bsz * SEQ_LEN
    micro_bsz=1,
    # defaults to the value of micro_num
    valid_micro_num=4,
    # defaults to 0, means disable evaluate
    valid_every=0,
    pack_sample_into_one=False,
    total_steps=50,
    skip_batches="",
    rampup_batch_size="",
    # Datasets with less than 50 rows will be discarded
    min_length=50,
    # train_folder=TRAIN_FOLDER,
    # valid_folder=None,
    empty_cache_and_diag_interval=200,
    diag_outlier_ratio=1.1,
    use_flash_style_data_format=False,
    break_mode="pad",
    use_packed_dataset=False
)

grad_scaler = dict(
    fp16=dict(
        # the initial loss scale, defaults to 2**16
        initial_scale=2**16,
        # the minimum loss scale, defaults to None
        min_scale=1,
        # the number of steps to increase loss scale when no overflow occurs
        growth_interval=1000,
    ),
    # the multiplication factor for increasing loss scale, defaults to 2
    growth_factor=2,
    # the multiplication factor for decreasing loss scale, defaults to 0.5
    backoff_factor=0.5,
    # the maximum loss scale, defaults to None
    max_scale=2**24,
    # the number of overflows before decreasing loss scale, defaults to 2
    hysteresis=2,
)

hybrid_zero_optimizer = dict(
    # Enable low_level_optimzer overlap_communication
    overlap_sync_grad=True,
    overlap_sync_param=False,
    # bucket size for nccl communication params
    reduce_bucket_size=512 * 1024 * 1024,
    # grad clipping
    clip_grad_norm=1.0,
)

loss = dict(
    label_smoothing=0,
)

adam = dict(
    lr=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=0.01,
)

lr_scheduler = dict(
    total_steps=data["total_steps"],
    init_steps=0,  # optimizer_warmup_step
    warmup_ratio=0.01,
    eta_min=1e-5,
    last_epoch=-1,
)

beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)

use_fp32_norm = False
model = dict(
    checkpoint=False,  # The proportion of layers for activation aheckpointing, the optional value are True/False/[0-1]
    num_attention_heads=NUM_ATTENTION_HEAD,
    embed_split_hidden=True,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=False,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    mlp_ratio=MLP_RATIO,
    apply_post_layer_norm=False,
    dtype="torch.float16",  # Support: "torch.float16", "torch.half", "torch.bfloat16", "torch.float32", "torch.tf32"
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-5,
    num_chunks=1,  # if num_chunks > 1, interleaved pipeline scheduler is used.
    # attn_type="npu_flash",
    use_flash_attn=False,
)
#npu_flash
if model_type == "LLAMA2" or model_type == "INTERNLM2_PUBLIC":
    model.update({'num_kv_attention_heads': NUM_KV_ATTENTION_HEAD})
    model.update({'no_bias': True})

parallel = dict(
    zero1=dict(size=1),
    tensor=dict(size=4, mode="mtp"),
    pipeline=dict(size=1, interleaved_overlap=True),
    weight=dict(size=1, overlap=False, memory_pool=False),
)

cudnn_deterministic = False
cudnn_benchmark = False

monitor = dict(
    # feishu alert configs
    alert=dict(
        enable_feishu_alert=DO_ALERT,
        feishu_alert_address=None,  # feishu webhook to send alert message
        light_monitor_address=None,  # light_monitor address to send heartbeat
        alert_file_path=f"llm_alter/{JOB_NAME}_alert.log",
    ),
    tensorboard=dict(
        queue_max_length=10,
    ),
)

# metric_dtype can be "fp32" or other string
# only when set to "fp32" will use fp32 to calc in metrics
# metric_dtype = "fp32"
