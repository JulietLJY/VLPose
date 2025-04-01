_base_ = ['vit-L.py']

find_unused_parameters=True
model = dict(
    text_encoder=dict(
        text_embed=768,  # depends on text encoder 
        image_embed=1024, # depends on image encoder 
        # name and model type choices:
        # albef_feature_extractor        base
        # blip_feature_extractor         base
        # clip_feature_extractor         ViT-B-32, ViT-B-16, ViT-L-14, ViT-L-14-336, RN50
        name='blip_feature_extractor',
        model_type='base',
        num_keywords=1,
    ),
    image_encoder=dict(
        type='ViTPrompt',
        frozen_stages=24,
        prompt_config=dict(
            deep=False, 
            num_tokens=10,
        ),
    ),
    image_text_matcher=dict(
        matcher_type='concat_attn',
    ),
    head=dict(
        deconv_type='first_amiddle_last_add',
    ),
)

