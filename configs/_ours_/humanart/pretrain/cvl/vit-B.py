# TODO: Change the pretrain path to the weight of the 

_base_ = ['../humanart.py']

# codec settings
codec = dict(
    type='UDPHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

data_root = 'data/'

# model settings
load_from=YOUR_COCO_PRETRAINED_PATH
model = dict(
    type='ConditionalVisionLanguageTopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    text_encoder=dict(
        text_prompt_file=data_root + 'HumanArt/category_prompt.yaml',
        keypoints_anno_file=data_root + 'HumanArt/keypoints.yaml',
    ),
    image_text_matcher=dict(
        # matcher type choices: ['attn', 'cosine']
        embed_dims=768, 
        num_heads=8, 
        qkv_bias=True, 
        attn_drop=0.1,
    ),
    image_encoder=dict(
        type='VisionTransformer',
        arch='base',
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.3,
        with_cls_token=False,
        output_cls_token=False,
        patch_cfg=dict(padding=2),
    ),
    head=dict(
        type='VisionLanguageHeatmapHead',
        freeze_main_branch=True,
        in_channels=768,
        out_channels=17,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

train_dataloader = dict(
    batch_size=64,
    num_workers=4,)
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
)