_base_ = [
    "../../_base_/models/htc_without_semantic_swin_fpn.py",
    "../../_base_/datasets/facsed_instance.py",
    "../../_base_/schedules/schedule_1x.py",
    "../../_base_/default_runtime.py",
]

model = dict(
    backbone=dict(
        type="CBSwinTransformer",
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=True,
    ),
    neck=dict(type="CBFPN", in_channels=[128, 256, 512, 1024]),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type="soft_nms"),
        )
    ),
)

data_root = "data/FACSED/"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# augmentation strategy originates from HTC
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="Resize",
        img_scale=[(1600, 400), (1600, 1400)],
        multiscale_mode="range",
        keep_ratio=True,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="SegRescale", scale_factor=1 / 8),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_masks"],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1600, 1400),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

samples_per_gpu = 1
data = dict(
    samples_per_gpu=samples_per_gpu,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.0001 * (samples_per_gpu / 2),
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

lr_config = dict(step=[16, 19])
runner = dict(type="EpochBasedRunnerAmp", max_epochs=20)
evaluation = dict(interval=20, metric="bbox")
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
