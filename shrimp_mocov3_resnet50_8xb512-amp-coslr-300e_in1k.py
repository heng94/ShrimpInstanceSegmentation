#-------------- model settings
temperature = 1.0
model = dict(
    type='MoCoV3',
    base_momentum=0.99,  # 0.99 for 100e and 300e, 0.996 for 1000e
    data_preprocessor=dict(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        with_bias=False,
        with_last_bn=True,
        with_last_bn_affine=False,
        with_last_bias=False,
        with_avg_pool=True,
        vit_backbone=False),
    head=dict(
        type='MoCoV3Head',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            num_layers=2,
            with_bias=False,
            with_last_bn=False,
            with_last_bn_affine=False,
            with_last_bias=False,
            with_avg_pool=False),
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=2 * temperature),
        temperature=temperature))


#-------------- optimizer settings
# 8*512: 4.8
# 2*16: 0.0375
optimizer = dict(type='LARS', lr=0.0375, weight_decay=1e-6)
optim_wrapper = dict(optimizer=optimizer)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=290,
        by_epoch=True,
        begin=10,
        end=300,
        convert_to_iter_based=True)
]


#-------------- dataset settings
dataset_type = 'mmcls.CustomDataset'
data_root = '/data_hdd1/ZhouHeng/Data/ShrimpSSL/data_v1/'
file_client_args = dict(backend='disk')
view_pipeline1 = [
    dict(
        type='RandomResizedCrop', size=480, scale=(0.2, 1.), backend='pillow'),
        # type='RandomResizedCrop', size=224, scale=(0.2, 1.), backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=1.),
    dict(type='RandomSolarize', prob=0.),
    dict(type='RandomFlip', prob=0.5),
]
view_pipeline2 = [
    dict(
        type='RandomResizedCrop', size=480, scale=(0.2, 1.), backend='pillow'),
        # type='RandomResizedCrop', size=224, scale=(0.2, 1.), backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.1),
    dict(type='RandomSolarize', prob=0.2),
    dict(type='RandomFlip', prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='MultiView',
        num_views=[1, 1],
        transforms=[view_pipeline1, view_pipeline2]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))


#-------------- runtime settings
default_scope = 'mmselfsup'
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)

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
            name='MoCov3_480_1',
            save_code=True,
            dir='/data_hdd1/ZhouHeng/Data/ShrimpSSL/ssl_result_size_480/MoCov3_480_1/',
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
load_from = '/data_hdd1/ZhouHeng/Data/ShrimpSSL/mocov3_resnet50_8xb512-amp-coslr-800e_in1k.pth'
resume = False
work_dir = '/data_hdd1/ZhouHeng/Data/ShrimpSSL/ssl_result_size_480/MoCov3_480_1/'
