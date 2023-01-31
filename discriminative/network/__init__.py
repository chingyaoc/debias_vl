import numpy as np
import torch
import torch.nn as nn

# Pretrained models
import network.clip as base_clip


def get_parameter_count(model, verbose=True):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    if verbose:
        print(f'-> Number of parameters: {num_params}')
    return num_params


def load_base_model(base_model_args, args, clip=None):
    """
    Load foundation model, foundation model transform, embedding functions
    Returns:
    - base_model, base_transform, get_embeddings, get_dataset_embeddings, get_zeroshot_predictions
    """
    if 'clip' in base_model_args:
        clip_name = base_model_args[1]  # example: 'RN50'
        base_model, transform = clip.load(clip_name)
        args.num_base_model_parameters = get_parameter_count(base_model, 
                                                             verbose=True)
        pipeline = None
        base_transform = transform
        get_embeddings = base_clip.get_embeddings
        get_dataset_embeddings = base_clip.get_dataset_embeddings
        get_zeroshot_predictions = base_clip.get_zeroshot_predictions
        
    elif 'cloob' in base_model_args:
        base_model, transform = load_cloob_model(args)
        args.num_base_model_parameters = get_parameter_count(base_model, 
                                                             verbose=True)
        pipeline = None
        base_transform = transform
        get_embeddings = base_clip.get_embeddings
        get_dataset_embeddings = base_clip.get_dataset_embeddings
        get_zeroshot_predictions = base_clip.get_zeroshot_predictions
    
        
    else:
        # Ex.) --load_base_model 'EleutherAI/gpt-neo-1.3B_cls'
        if 'cls' in base_model_args:
            args.sequence_classification_model = True
            
        base_model, transform, tokenizer = base_lm.load_pretrained_language_model(
            args.sequence_classification_model, args
        )
        args.num_base_model_parameters = get_parameter_count(base_model, 
                                                             verbose=True)
        # Only use base model for feature extraction for now
        device_id = (torch.cuda.current_device() 
                     if torch.cuda.is_available() and not args.no_cuda else -1)
        pipeline = base_lm.load_pipeline(base_model, tokenizer,
                                         args.max_token_length,
                                         device=device_id,
                                         task='feature-extraction')
        base_model = pipeline
        base_transform = None
        get_embeddings = base_lm.get_embeddings
        get_dataset_embeddings = base_lm.get_dataset_embeddings
        get_zeroshot_predictions = base_lm.get_zeroshot_predictions
    return base_model, base_transform, get_embeddings, get_dataset_embeddings, get_zeroshot_predictions



"""
Model attributes, from https://github.com/kohpangwei/group_DRO/blob/master/models.py

Used for: Waterbirds
"""

model_attributes = {
    'bert': {
        'feature_type': 'text'
    },
    'inception_v3': {
        'feature_type': 'image',
        'target_resolution': (299, 299),
        'flatten': False
    },
    'wideresnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet34': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': False
    },
    'resnet18': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'raw_logistic_regression': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': True,
    },  # CLIP input resolutions
    'RN50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'RN18': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'RN101': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'RN50x16': {
        'feature_type': 'image',
        'target_resolution': (384, 384),
        'flatten': False
    },
    'ViTB32': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'ViTB16': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    }
}
