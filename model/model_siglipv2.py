import torch
import torch.nn as nn
from transformers import AutoModelForVision

MODEL_NAME = "google/siglip2-base-patch16-224"

class CustomModel(nn.Module):
    def __init__(self, num_classes: int, extra_dim: int = 0, model_name: str = MODEL_NAME):
        super().__init__()
        self.model_base = AutoModelForVision.from_pretrained(model_name)
        in_features = self.model_base.config.vision_config.hidden_size

        self.extra_dim = extra_dim
        if extra_dim > 0:
            self.extra_proj = nn.Sequential(
                nn.BatchNorm1d(extra_dim),
                nn.ReLU(inplace=True)
            )
            self.in_features = in_features + extra_dim
        else:
            self.extra_proj = None
            self.in_features = in_features

        self.classifier = nn.Linear(self.in_features, num_classes)

    def forward(self, pixel_values: torch.Tensor, extra_vec: torch.Tensor = None):
        outputs = self.model_base(pixel_values=pixel_values)
        feat = outputs.image_embeds  # (B, in_features)

        if self.extra_proj is not None and extra_vec is not None:
            extra_feat = self.extra_proj(extra_vec)       # (B, in_features)
            feat = torch.cat([feat, extra_feat], dim=1) 

        return self.classifier(feat)


def build_model(num_classes: int, extra_dim: int = 0, model_name: str = MODEL_NAME):
    return CustomModel(num_classes=num_classes, extra_dim=extra_dim, model_name=model_name)
