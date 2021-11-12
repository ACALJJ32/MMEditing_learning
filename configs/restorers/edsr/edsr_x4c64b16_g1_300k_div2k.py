exp_name = 'edsr_x4c64b16_g1_300k_div2k'

scale = 4
# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='EDSR',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=16,
        upscale_factor=scale,
        res_scale=1,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1.0, 1.0, 1.0)),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)

# dataset settings
train_dataset_type = 'SRREDSDataset'
val_dataset_type = 'SRREDSDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='PairedRandomCrop', gt_patch_size=196),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='/media/test/Disk2/DATA/VSR/Tencent_SDR/train/SDR_540p_train_frames',
            gt_folder='/media/test/Disk2/DATA/VSR/Tencent_SDR/train/SDR_4K_train_frames',
            ann_file='/media/test/Disk2/DATA/VSR/Tencent_SDR/train/meta_info_GKY_GT.txt',
            num_input_frames=5,
            pipeline=train_pipeline,
            scale=4,
            val_partition='REDS4',
            test_mode=False)),
    val=dict(
        type=val_dataset_type,
        lq_folder='/media/test/Disk2/DATA/VSR/Tencent_SDR/train/SDR_540p_train_frames',
        gt_folder='/media/test/Disk2/DATA/VSR/Tencent_SDR/train/SDR_4K_train_frames',
        ann_file='/media/test/Disk2/DATA/VSR/Tencent_SDR/train/meta_info_GKY_GT.txt',
        num_input_frames=5,
        pipeline=test_pipeline,
        scale=4,
        val_partition='REDS4',
        test_mode=True),
    test=dict(
        type=val_dataset_type,
        lq_folder='/media/test/Disk2/DATA/VSR/Tencent_SDR/train/SDR_540p_train_frames',
        gt_folder='/media/test/Disk2/DATA/VSR/Tencent_SDR/train/SDR_4K_train_frames',
        ann_file='/media/test/Disk2/DATA/VSR/Tencent_SDR/train/meta_info_GKY_GT.txt',
        num_input_frames=5,
        pipeline=test_pipeline,
        scale=4,
        val_partition='REDS4',
        test_mode=True),
)

# data = dict(
#     workers_per_gpu=8,
#     train_dataloader=dict(samples_per_gpu=16, drop_last=True),
#     val_dataloader=dict(samples_per_gpu=1),
#     test_dataloader=dict(samples_per_gpu=1),
#     train=dict(
#         type='RepeatDataset',
#         times=1000,
#         dataset=dict(
#             type=train_dataset_type,
#             lq_folder='/media/test/Disk2/DATA/VSR/Tencent_SDR/train/SDR_540p_train_frames',
#             gt_folder='/media/test/Disk2/DATA/VSR/Tencent_SDR/train/SDR_4K_train_frames',
#             ann_file='/media/test/Disk2/DATA/VSR/Tencent_SDR/train/meta_info_GKY_GT.txt',
#             pipeline=train_pipeline,
#             scale=scale)),
#     val=dict(
#         type=val_dataset_type,
#         lq_folder='/media/test/Disk2/DATA/VSR/REDS/val/val_sharp_bicubic/X4/000',
#         gt_folder='/media/test/Disk2/DATA/VSR/REDS/val/val_sharp/000',
#         pipeline=test_pipeline,
#         scale=scale,
#         filename_tmpl='{}'),
#     test=dict(
#         type=val_dataset_type,
#         lq_folder='/media/test/Disk2/DATA/VSR/REDS/val/val_sharp_bicubic/X4/000',
#         gt_folder='/media/test/Disk2/DATA/VSR/REDS/val/sharp/000',
#         pipeline=test_pipeline,
#         scale=scale,
#         filename_tmpl='{}'))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = 300000
lr_config = dict(policy='Step', by_epoch=False, step=[200000], gamma=0.5)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=5000, save_image=True, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit-sr'))
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = 'weight/edsr/edsr_x4c64b16_1x16_300k_div2k_20200608-3c2af8a3.pth'
resume_from = None
workflow = [('train', 1)]
