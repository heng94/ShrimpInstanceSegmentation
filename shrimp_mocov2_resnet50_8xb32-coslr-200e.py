#-------------- model settings
model = dict(
    type='MoCo',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    data_preprocessor=dict(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='MoCoV2Neck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='mmcls.CrossEntropyLoss'),
        temperature=0.2))


#-------------- optimizer settings
# 8*32: 0.03
# 2*32: 0.0075
optimizer = dict(type='SGD', lr=0.0075, weight_decay=1e-4, momentum=0.9)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
# learning rate scheduler
# param_scheduler = [dict(type='CosineAnnealingLR', T_max=200, by_epoch=True, begin=0, end=200)]
param_scheduler = [dict(type='MultiStepLR', by_epoch=True, milestones=[120, 160], gamma=0.1)]


#-------------- dataset settings
dataset_type = 'mmcls.CustomDataset'
data_root = '/data_hdd1/ZhouHeng/Data/ShrimpSSL/data_v1/'
file_client_args = dict(backend='disk')

# The difference between mocov2 and mocov1 is the transforms in the pipeline
view_pipeline = [
    # dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.), backend='pillow'),
    dict(type='RandomResizedCrop', size=480, scale=(0.2, 1.), backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
        prob=0.8,
    ),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989),
    ),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.5),
    dict(type='RandomFlip', prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline,
    ),
)


#-------------- runtime settings
default_scope = 'mmselfsup'
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=50, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

vis_backends = [
    dict(
        type='WandbVisBackend', 
        init_kwargs=dict(
            project='mmselfsup',
            name='MoCov2_480_1',
            save_code=True,
            dir='/data_hdd1/ZhouHeng/Data/ShrimpSSL/ssl_result_size_480/MoCov2_480_1/',
            magic=True,
        ),
    ),
]

visualizer = dict(
    type='SelfSupVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(
    window_size=10,
    custom_cfg=[dict(data_src='', method='mean', window_size='global')]
)

log_level = 'INFO'
load_from = '/data_hdd1/ZhouHeng/Data/ShrimpSSL/mocov2_resnet50_8xb32-coslr-200e_in1k.pth'
resume = False
work_dir = '/data_hdd1/ZhouHeng/Data/ShrimpSSL/ssl_result_size_480/MoCov2_480_1/'