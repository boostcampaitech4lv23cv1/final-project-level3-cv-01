import datetime
from pytz import timezone

now = datetime.datetime.now(timezone("Asia/Seoul")).strftime("_%y%m%d_%H%M%S")

checkpoint_config = dict(interval=100)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
        dict(
            type="WandbLoggerHook",
            interval=50,
            init_kwargs=dict(
                project="final_project_mmpose",
                entity="kidsarebornstars",
                name="hrnet_w48_udp" + now,
            ),
        ),
    ],
)


log_level = "INFO"
load_from = None
resume_from = None
dist_params = dict(backend="nccl")
workflow = [("train", 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = "fork"
