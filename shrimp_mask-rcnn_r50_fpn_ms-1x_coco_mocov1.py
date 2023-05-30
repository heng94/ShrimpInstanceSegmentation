_base_ = 'mmdet::mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
#--------------- model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(frozen_stages=-1, norm_cfg=norm_cfg, norm_eval=False),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            num_classes=1,
            norm_cfg=norm_cfg),
        mask_head=dict(
            num_classes=1,
            norm_cfg=norm_cfg),
        ),
    )


#--------------- dataset settings
file_client_args = dict(backend='disk')

dataset_type = 'CocoDataset'
data_root = '/data_hdd1/ZhouHeng/Data/ShrimpInstanceSeg/'

metainfo = {
  'classes': ('shrimp', ), 
  'palette': ['coco',]
}

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train_600.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val_200.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val_200.json',
    metric=['bbox', 'segm'],
    format_only=False,
)

test_evaluator = val_evaluator


#--------------- optimizer settings
# learning rate

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=96,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

# optimizer
# base_batch_size: 8xb2, lr: 0.02
# new lr: 1xb8 ==> 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
)
auto_scale_lr = dict(enable=False, base_batch_size=16)


#--------------- running settings
# one period: 12 epochs
default_scope = 'mmdet'
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=96, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [
    dict(
        type='WandbVisBackend', 
        init_kwargs=dict(
            project='mmselfseg',
            name='MoCov1_224_120',
            save_code=True,
            dir='/data_hdd1/ZhouHeng/Data/InstanceSeg/MoCov1_224_120/',
            magic=True,
        ),
    ),
]

visualizer = dict(
    type='DetLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer',
)
log_processor = dict(type='LogProcessor', window_size=1, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
