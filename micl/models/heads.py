from typing import Optional, Tuple

import torch
import torch.nn as nn


class BoxHead(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int = 4):
        super().__init__()
        self.dense0 = nn.Linear(hidden_size, hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(hidden_size, out_dim)

    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor:
        output = self.dense0(image_features)
        output = self.gelu(output)
        output = self.dense1(output)
        output = self.gelu(output)
        output = self.dense2(output)
        return output


class ClassHead(nn.Module):
    def __init__(
            self,
            hidden_size_image: int,
            hidden_size_text: int,
            softmax_scale: Optional[float] = None,
            hidden_size: int = 512,
            learnable_scale: bool = False
    ):
        super().__init__()
        # self.image_to_query = nn.Linear(hidden_size_image, hidden_size_text)
        self.image_projector = nn.Linear(hidden_size_image, hidden_size)
        self.text_projector = nn.Linear(hidden_size_text, hidden_size)
        if softmax_scale is None:
            self.logit_shift = nn.Linear(hidden_size_image, 1)
            self.logit_scale = nn.Linear(hidden_size_image, 1)
        elif learnable_scale:
          self.logit_scale = nn.Parameter(torch.tensor(softmax_scale))
        self.learnable_scale = learnable_scale
        self.elu = nn.ELU()
        self.softmax_scale = softmax_scale

    def forward(
        self,
        image_embeds: torch.FloatTensor,
        query_embeds: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor]:
        # image_class_embeds = self.image_to_query(image_embeds)
        image_class_embeds = self.image_projector(image_embeds)
        query_embeds = self.text_projector(query_embeds)
        # Normalize image and text features
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)

        # Get class predictions
        pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)

        if self.softmax_scale is None:
        # Apply a learnable shift and scale to logits
            logit_scale = self.logit_scale(image_embeds)
            logit_scale = self.elu(logit_scale) + 1
            logit_shift = self.logit_shift(image_embeds)
            pred_logits = (pred_logits + logit_shift) * logit_scale
        elif self.learnable_scale: 
            pred_logits = pred_logits * self.logit_scale.exp()
        else:
            pred_logits = pred_logits * self.softmax_scale
        return pred_logits
