import torch
import torch.nn as nn
from transformers import SiglipVisionModel
from build_projector import build_vision_projector
MODEL_NAME = "google/siglip2-base-patch16-224"

class CustomModel(nn.Module):
    def __init__(self, num_classes: int, extra_dim: int = 0, model_name: str = MODEL_NAME, projector_type: str = "mlp2x_gelu"):
        super().__init__()
        # Backbone SigLIP (vision-only)
        self.model_base = SiglipVisionModel.from_pretrained(model_name)
        in_features = self.model_base.config.hidden_size  # hidden dim

        self.extra_dim = extra_dim
        if extra_dim > 0:
            self.extra_proj = build_vision_projector(
                mm_hidden_size=extra_dim,
                hidden_size= in_features,
                projector_type= projector_type,
            )
            # self.extra_proj = nn.Sequential(
            #     nn.BatchNorm1d(extra_dim),
            #     nn.ReLU(inplace=True)
            # )
            self.in_features = in_features * 2
        else:
            self.extra_proj = None
            self.in_features = in_features

        self.classifier = nn.Linear(self.in_features, num_classes)

    def forward(self, pixel_values: torch.Tensor, extra_vec: torch.Tensor = None):
        outputs = self.model_base(pixel_values=pixel_values)
        feat = outputs.pooler_output  # (B, hidden_size)

        if self.extra_proj is not None and extra_vec is not None:
            extra_feat = self.extra_proj(extra_vec)        # (B, hidden_size)
            feat = torch.cat([feat, extra_feat], dim=1)    # (B, 2*hidden_size)

        return self.classifier(feat)


def build_model(
    num_classes: int, extra_dim: int = 0, model_name: str = MODEL_NAME, projector_type: str = "mlp2x_gelu", pretrained: bool = True,):
    return CustomModel(
        num_classes=num_classes,
        extra_dim=extra_dim,
        model_name=model_name,
        projector_type=projector_type,
        pretrained=pretrained,
    )
