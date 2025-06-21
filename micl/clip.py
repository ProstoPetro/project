from functools import cached_property
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPVisionConfig,
    CLIPVisionModel,
    PreTrainedTokenizerFast,
)

from .image_processor import ImageProcessor


class CLIPStuff:
    def __init__(self, model_dirpath=None):
        if model_dirpath is None:
            model_dirpath = Path('/home/jovyan/models/mimiccxr-clip-vit-large-patch14-336')
        self.model_dirpath = model_dirpath
        
    @cached_property
    def image_processor(self):
        return ImageProcessor()
    
    @cached_property
    def vision_config(self):
        return CLIPVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            image_size=336,
            patch_size=14,
        )

    @cached_property
    def text_config(self):
        return CLIPTextConfig(
            vocab_size=30_000,
            max_position_embeddings=128,
            eos_token_id=1
        )
    
    @cached_property
    def tokenizer(self):
        return PreTrainedTokenizerFast(
            bos_token='[BOS]',
            eos_token='[EOS]',
            unk_token='[UNK]',
            sep_token='[SEP]',
            pad_token='[PAD]',
            cls_token='[CLS]',
            mask_token='[MASK]',
            tokenizer_file=str(self.model_dirpath / 'tokenizer.json')
        )


def mimiccxr_clip() -> CLIPTextModel:
    image_processor = ImageProcessor()

    # FIXME: download weights from hf
    model_dirpath = Path('/home/jovyan/models/mimiccxr-clip-vit-large-patch14-336')

    config = CLIPVisionConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        image_size=336,
        patch_size=14,
    )
    vision_model = CLIPVisionModel(config)
    vision_model.load_state_dict(torch.load(model_dirpath / 'vision_model.pt'))

    vision_projector = nn.Linear(1024, 768, bias=False)
    vision_projector.load_state_dict(torch.load(model_dirpath / 'vision_projector.pt'))

    tokenizer = PreTrainedTokenizerFast(
        bos_token='[BOS]',
        eos_token='[EOS]',
        unk_token='[UNK]',
        sep_token='[SEP]',
        pad_token='[PAD]',
        cls_token='[CLS]',
        mask_token='[MASK]',
        tokenizer_file=str(model_dirpath / 'tokenizer.json')
    )

    config = CLIPTextConfig(
        vocab_size=30_000,
        max_position_embeddings=128,
        eos_token_id=1
    )
    text_model = CLIPTextModel(config=config)
    text_model.load_state_dict(torch.load(model_dirpath / 'text_model.pt'))

    text_projector = nn.Linear(512, 768, bias=False)
    text_projector.load_state_dict(torch.load(model_dirpath / 'text_projector.pt'))

    return (
        image_processor, vision_model, vision_projector,
        tokenizer, text_model, text_projector
    )
