exp_name = 'egovsr'

scale = 4

# model settings
model = dict(
    type='RealEGOVSR',
    generator=dict(
        type='RealEGOVSRNet',
        mid_channels=64,
        num_propagation_blocks=20,
        num_cleaning_blocks=20,
        dynamic_refine_thres=255,  # change to 5 for test
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth',
        is_fix_cleaning=False,
        is_sequential_cleaning=False),
    discriminator=dict(
        type='UNetDiscriminatorWithSpectralNorm',
        in_channels=3,
        mid_channels=64,
        skip_connection=True),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    cleaning_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    mask_loss=dict(type='MaskLoss', loss_weight=0.2, k=100),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={
            '2': 0.1,
            '7': 0.1,
            '16': 1.0,
            '25': 1.0,
            '34': 1.0,
        },
        vgg_type='vgg19',
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=5e-2,
        real_label_val=1.0,
        fake_label_val=0),
    is_use_sharpened_gt_in_pixel=True,
    is_use_sharpened_gt_in_percep=True,
    is_use_sharpened_gt_in_gan=False,
    is_use_ema=True,
)

# model training and testing settings
train_cfg = dict()
test_cfg = dict(metrics=['PSNR'],input_order='HWC', crop_border=0, convert_to='y')  # change to [] for test

# dataset settings
train_dataset_type = 'SRFolderMultipleGTDataset'
val_dataset_type = 'SRVimeo90KDataset'
train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='FixedCrop', keys=['gt','lq'], crop_size=(256, 256)),
    dict(type='RescaleToZeroOne', keys=['gt','lq']),
    dict(type='Flip', keys=['gt','lq'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['gt','lq'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['gt','lq'], transpose_ratio=0.5),
    dict(type='MirrorSequence', keys=['gt','lq']),
    dict(type='CopyValues', src_keys=['lq'], dst_keys=['lq_clean']),
    dict(
        type='UnsharpMasking',
        keys=['gt'],
        kernel_size=51,
        sigma=0,
        weight=0.5,
        threshold=10),
    #dict(type='CopyValues', src_keys=['gt_unsharp'], dst_keys=['lq']),
    dict(
        type='RandomVideoBlur',
        params=dict(
            frame=[7, 5, 3, 1],
            frame_prob=[0.2, 0.35, 0.35, 0.1],
            prob=1),
        keys=['lq'],
    ),
    dict(
        type='RandomBlur',
        params=dict(
            kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
            kernel_list=[
                'iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                'plateau_iso', 'plateau_aniso', 'sinc'
            ],
            kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
            sigma_x=[0.2, 3],
            sigma_y=[0.2, 3],
            rotate_angle=[-3.1416, 3.1416],
            beta_gaussian=[0.5, 4],
            beta_plateau=[1, 2],
            sigma_x_step=0.02,
            sigma_y_step=0.02,
            rotate_angle_step=0.31416,
            beta_gaussian_step=0.05,
            beta_plateau_step=0.1,
            omega_step=0.0628),
        keys=['lq'],
    ),
    dict(
        type='RandomResize',
        params=dict(
            resize_mode_prob=[0.2, 0.7, 0.1],  # up, down, keep
            resize_scale=[0.15, 1.5],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3.0, 1 / 3.0, 1 / 3.0],
            resize_step=0.015,
            is_size_even=True),
        keys=['lq'],
    ),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian', 'poisson'],
            noise_prob=[0.5, 0.5],
            gaussian_sigma=[1, 30],
            gaussian_gray_noise_prob=0.4,
            poisson_scale=[0.05, 3],
            poisson_gray_noise_prob=0.4,
            gaussian_sigma_step=0.1,
            poisson_scale_step=0.005),
        keys=['lq'],
    ),
    dict(
        type='RandomJPEGCompression',
        params=dict(quality=[30, 95], quality_step=3),
        keys=['lq'],
    ),
    dict(
        type='RandomVideoCompression',
        params=dict(
            codec=['libx264', 'h264', 'mpeg4'],
            codec_prob=[1 / 3., 1 / 3., 1 / 3.],
            bitrate=[1e4, 1e5]),
        keys=['lq'],
    ),
    dict(
            type='RandomBlur',
            params=dict(
            prob=0.8,
            kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
            kernel_list=[
                'iso', 'aniso', 'generalized_iso', 'generalized_aniso',
                'plateau_iso', 'plateau_aniso', 'sinc'
            ],
            kernel_prob=[0.405, 0.225, 0.108, 0.027, 0.108, 0.027, 0.1],
            sigma_x=[0.2, 1.5],
            sigma_y=[0.2, 1.5],
            rotate_angle=[-3.1416, 3.1416],
            beta_gaussian=[0.5, 4],
            beta_plateau=[1, 2],
            sigma_x_step=0.02,
            sigma_y_step=0.02,
            rotate_angle_step=0.31416,
            beta_gaussian_step=0.05,
            beta_plateau_step=0.1,
            omega_step=0.0628),
        keys=['lq'],
    ),
    dict(
        type='RandomResize',
        params=dict(
            resize_mode_prob=[0.3, 0.4, 0.3],  # up, down, keep
            resize_scale=[0.3, 1.2],
            resize_opt=['bilinear', 'area', 'bicubic'],
            resize_prob=[1 / 3., 1 / 3., 1 / 3.],
            resize_step=0.03,
            is_size_even=True),
        keys=['lq'],
    ),
    dict(
        type='RandomNoise',
        params=dict(
            noise_type=['gaussian', 'poisson'],
            noise_prob=[0.5, 0.5],
            gaussian_sigma=[1, 25],
            gaussian_gray_noise_prob=0.4,
            poisson_scale=[0.05, 2.5],
            poisson_gray_noise_prob=0.4,
            gaussian_sigma_step=0.1,
            poisson_scale_step=0.005),
        keys=['lq'],
    ),
    dict(
        type='RandomJPEGCompression',
        params=dict(quality=[30, 95], quality_step=3),
        keys=['lq'],
    ),
    dict(
        type='DegradationsWithShuffle',
        degradations=[
            dict(
                type='RandomVideoCompression',
                params=dict(
                    codec=['libx264', 'h264', 'mpeg4'],
                    codec_prob=[1 / 3., 1 / 3., 1 / 3.],
                    bitrate=[1e4, 1e5]),
                keys=['lq'],
            ),
            [
                dict(
                    type='RandomResize',
                    params=dict(
                        target_size=(64, 64),
                        resize_opt=['bilinear', 'area', 'bicubic'],
                        resize_prob=[1 / 3., 1 / 3., 1 / 3.]),
                ),
                dict(
                    type='RandomBlur',
                    params=dict(
                        prob=0.8,
                        kernel_size=[7, 9, 11, 13, 15, 17, 19, 21],
                        kernel_list=['sinc'],
                        kernel_prob=[1],
                        omega=[3.1416 / 3, 3.1416],
                        omega_step=0.0628),
                ),
            ]
        ],
        keys=['lq'],
    ),

    dict(type='Quantize', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq', 'gt', 'gt_unsharp', 'lq_clean']),
    dict(
        type='Collect', keys=['lq', 'gt', 'gt_unsharp', 'lq_clean'], meta_keys=['gt_path'])
]

val_pipeline = [
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='MirrorSequence', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

test_pipeline = [
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='MirrorSequence', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    # dict(
    #    type='MATLABLikeResize',scale=1/4,
    #    keys=['lq'],
    # ),
    dict(type='RescaleToZeroOne', keys=['lq']),
    #dict(type='MirrorSequence', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
   train=dict(
        type='RepeatDataset',
        times=150,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='/media/data4/dataset/reds/train/train_blur', #using train_sharp folder for first-order training
            gt_folder='/media/data4/dataset/reds/train/train_sharp',
            num_input_frames=15,
            pipeline=train_pipeline,
            scale=4,
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
            lq_folder='//home/chiyc/Data/VSR/dataset/EGOVSR_dataset/valid/lr',
            gt_folder='/home/chiyc/Data/VSR/dataset/EGOVSR_dataset/valid/hr',
            ann_file='/home/chiyc/Data/VSR/dataset/EGOVSR_dataset/valid_anno_fast.txt',
        pipeline=val_pipeline,
        scale=4,
        num_input_frames=7,
        test_mode=True),
    # test
    test=dict(
        type=val_dataset_type,
            lq_folder='//home/chiyc/Data/VSR/dataset/EGOVSR_dataset/test/lr',
            gt_folder='/home/chiyc/Data/VSR/dataset/EGOVSR_dataset/test/hr',
            ann_file='/home/chiyc/Data/VSR/dataset/EGOVSR_dataset/test_anno_fast.txt',
        pipeline=val_pipeline,
        scale=4,
        num_input_frames=7,
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(type='Adam', lr=5e-5, betas=(0.9, 0.99)),
    discriminator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)))

# learning policy
total_iters = 250000
lr_config = dict(policy='Step', by_epoch=False, step=[400000], gamma=1)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)

# remove gpu_collect=True in non distributed training
evaluation = dict(interval=5000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
visual_config = None

# custom hook
custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=1,
        interp_cfg=dict(momentum=0.999),
    )
]

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./experiments/{exp_name}'
load_from = None#'/home/chiyc/Data/VSR/mmediting/experiments/egovsr-v3-L1/latest.pth'
resume_from = None#'/home/chiyc/Data/VSR/mmediting/experiments/egovsr-v3/latest.pth'
workflow = [('train', 1)]
