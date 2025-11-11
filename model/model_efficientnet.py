import timm
import torch.nn as nn
import torch
from build_projector import build_vision_projector

MODEL_NAME = "efficientnet_b1.ft_in1k"

class CustomModel(nn.Module):
    def __init__(self, num_classes: int, extra_dim: int = 0, pretrained: bool = True,projector_type: str = "mlp2x_gelu"):
        super().__init__()
        # Backbone EfficientNet
        self.model_base = timm.create_model(MODEL_NAME, pretrained=pretrained)
        in_features = self.model_base.classifier.in_features
        self.model_base.classifier = nn.Identity()

        self.backbone_dim = in_features
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

    def forward(self, x, extra_vec=None):
        feat = self.model_base(x)  # (B, in_features)

        if self.extra_proj is not None and extra_vec is not None:
            extra_feat = self.extra_proj(extra_vec)  # (B, in_features)
            feat = torch.cat([feat, extra_feat], dim=1) 

        out = self.classifier(feat)
        return out


def build_model(num_classes: int, extra_dim: int = 0, pretrained: bool = True, projector_type: str = "mlp2x_gelu"):
    """
    Build EfficientNet model + optional extra vector.
    """
    return CustomModel(num_classes=num_classes, extra_dim=extra_dim, pretrained=pretrained, projector_type = projector_type)
