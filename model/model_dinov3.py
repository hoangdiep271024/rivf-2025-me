import torch
import torch.nn as nn
from transformers import AutoModel
from build_projector import build_vision_projector
MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

class DinoBackbone(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME, freeze: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 768))

        self.freeze = freeze
        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                out = self.model(pixel_values=pixel_values)
        else:
            out = self.model(pixel_values=pixel_values)

        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]  
        return pooled  # (B, hidden_size)


class CustomModel(nn.Module):
    def __init__(self, num_classes: int, extra_dim: int = 0,
                 pretrained: bool = True, 
                 freeze_backbone: bool = True,
                 model_name: str = MODEL_NAME,
                 projector_type: str = "mlp2x_gelu"):
        super().__init__()

        # Backbone DINOv3
        self.model_base = DinoBackbone(model_name=model_name, freeze=freeze_backbone)
        in_features = self.model_base.hidden_size 

        self.extra_dim = extra_dim
        if extra_dim > 0:
            # self.extra_proj = build_vision_projector(
            #     mm_hidden_size=extra_dim,
            #     hidden_size= in_features,
            #     projector_type= projector_type,
            # )
            self.extra_proj = nn.Sequential(
                nn.BatchNorm1d(extra_dim),
                nn.ReLU(inplace=True)
            )
            self.in_features = in_features + extra_dim 
        else:
            self.extra_proj = None
            self.in_features = in_features

        self.classifier = nn.Linear(self.in_features, num_classes)

    def extract_backbone_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Trả feature từ backbone: (B, in_features)."""
        return self.model_base(pixel_values)  # (B, 768)

    def forward(self, pixel_values: torch.Tensor, extra_vec: torch.Tensor = None,
                return_features: bool = False):
        feat = self.extract_backbone_feature(pixel_values)  # (B, 768)

        if self.extra_proj is not None and extra_vec is not None:
            extra_feat = self.extra_proj(extra_vec)          # (B, 768)
            feat = torch.cat([feat, extra_feat], dim=1)      # (B, 1536)

        if return_features:
            return feat  # fused feature

        out = self.classifier(feat)
        return out


def build_model(num_classes: int, extra_dim: int = 0,
                pretrained: bool = True,
                freeze_backbone: bool = True,
                model_name: str = MODEL_NAME,
                projector_type: str = "mlp2x_gelu"):
    return CustomModel(
        num_classes=num_classes,
        extra_dim=extra_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        model_name=model_name,
        projector_type = projector_type
    )
