from functools import lru_cache
from os import PathLike
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPVisionConfig,
    CLIPVisionModel,
    OwlViTForObjectDetection,
    OwlViTProcessor,
    CLIPModel
)

from .heads import BoxHead, ClassHead


class Detector(nn.Module):
    def __init__(
        self,
        vision_model: nn.Module,
        text_model: nn.Module,
        vision_config: CLIPVisionConfig,
        text_config: CLIPTextConfig,
        pool_size: int = 1,
        softmax_scale: Optional[float] = None,
        learnable_scale: bool = False,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.num_patches_height = self.num_patches_width = vision_config.image_size // vision_config.patch_size
        self.patch_size = vision_config.patch_size
        self.config = vision_config
        self.box_bias = self.compute_box_bias(self.num_patches_height // pool_size, self.num_patches_width // pool_size)
        
        self.layer_norm = nn.LayerNorm(vision_config.hidden_size, eps=vision_config.layer_norm_eps)
        self.sigmoid = nn.Sigmoid()
        self.clf_head = ClassHead(vision_config.hidden_size, text_config.hidden_size, softmax_scale=softmax_scale, learnable_scale=learnable_scale)
        self.box_head = BoxHead(vision_config.hidden_size)
        assert self.num_patches_height % pool_size == 0
        assert self.num_patches_width % pool_size == 0
        if pool_size != 1:
            self.pooling = nn.AvgPool2d(pool_size)
        else:
            self.pooling = None

    def get_image_embeddings(self, image: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=image)
        last_hidden = outputs.last_hidden_state         # (B, 1 + N, D)
        cls_embed  = outputs.pooler_output.unsqueeze(1) # (B, 1, D)

        # Оставляем только патч-токены
        patch_embeds = last_hidden[:, 1:, :]            # (B, N, D)

        # По желанию «склеиваем» с CLS-токеном как раньше
        class_token_out = cls_embed.expand_as(patch_embeds)  # (B, N, D)
        fused = self.layer_norm(patch_embeds * class_token_out)

        # Приводим к форме (B, N, D) или (B, n_boxes, D) после pooling
        B, N, D = fused.shape
        H = W = self.config.image_size // self.config.patch_size
        x = fused.view(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        if self.pooling:
            x = self.pooling(x)
            x = x.flatten(2).permute(0, 2, 1)           # (B, N', D)
        else:
            x = x.flatten(2).permute(0, 2, 1)           # (B, N, D)

        return x  # (B, 1369, D)
    
    def get_text_embeddings(self, text_tokens: torch.FloatTensor, attention_masks: torch.Tensor) -> torch.FloatTensor:
        return self.text_model(text_tokens, attention_masks).pooler_output # (M, text_hidden_state)
    
    def get_boxes(self, image_embeds):
        boxes = self.box_head(image_embeds)                       # (N, n_boxes or n_patches)
        boxes += self.box_bias.to(boxes.device)
        boxes = self.sigmoid(boxes)                               # (N, n_boxes, 4) boxes are cxcywh
        return boxes
    
    def get_logits(self, image_embeds, text_embeds):
        logits = self.clf_head(image_embeds, text_embeds)            # (N, n_boxes, M)
        return logits

    def forward(
        self,
        image: torch.FloatTensor,
        text_tokens: torch.FloatTensor,
        attention_masks: torch.Tensor,
    ):
        image_embeds = self.get_image_embeddings(image)                       # (N, n_boxes or n_patches, image_hidden_size)
        text_embeds = self.get_text_embeddings(text_tokens, attention_masks)  # (M, text_hidden_size)
        return self.get_boxes(image_embeds), self.get_logits(image_embeds, text_embeds)
    
    @staticmethod
    def normalize_grid_corner_coordinates(num_patches_height: int, num_patches_width: int) -> torch.Tensor: # копипаста из transformers
        # Create grid coordinates using torch
        x_coordinates = torch.arange(1, num_patches_width + 1, dtype=torch.float32)
        y_coordinates = torch.arange(1, num_patches_height + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x_coordinates, y_coordinates, indexing="xy")

        # Stack the coordinates and divide by their respective patch counts
        box_coordinates = torch.stack((xx, yy), dim=-1)
        box_coordinates[..., 0] /= num_patches_width
        box_coordinates[..., 1] /= num_patches_height

        # Flatten (h, w, 2) -> (h*w, 2)
        box_coordinates = box_coordinates.view(-1, 2)

        return box_coordinates

    @lru_cache(maxsize=2)
    def compute_box_bias( # копипаста из transformers
        self, num_patches_height: int, num_patches_width: int, feature_map: Optional[torch.FloatTensor] = None
    ) -> torch.Tensor:
        if feature_map is not None:
            raise ValueError("feature_map has been deprecated as an input. Please pass in num_patches instead")
        # The box center is biased to its position on the feature grid
        box_coordinates = self.normalize_grid_corner_coordinates(num_patches_height, num_patches_width)
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

        # Unnormalize xy
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

        # The box size is biased to the patch size
        box_size = torch.full_like(box_coord_bias, 1.0)
        box_size[..., 0] /= num_patches_width
        box_size[..., 1] /= num_patches_height
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

        # Compute box bias
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias
