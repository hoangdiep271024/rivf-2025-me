import torch
import torch.nn as nn
from transformers import AutoModel
hf_repo = "nvidia/C-RADIOv4-SO400M"
from build_projector import build_vision_projector
class CustomModel(nn.Module):
    def __init__(self, num_classes: int, extra_dim: int = 0, pretrained: bool = True, projector_type: str = "mlp2x_gelu"):
        super().__init__()
        # Backbone RADIO
        self.model_base = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
        in_features = 2560  

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

    def forward(self, x, extra_vec=None):
        out = self.model_base(x)
        feat = out[0] if isinstance(out, (tuple, list)) else out  # (B, 2304)

        if self.extra_proj is not None and extra_vec is not None:
            extra_feat = self.extra_proj(extra_vec)  # (B, 2304)
            feat = torch.cat([feat, extra_feat], dim=1)  

        return self.classifier(feat)


def build_model(num_classes: int, extra_dim: int = 0, pretrained: bool = True, projector_type: str = "mlp2x_gelu"):
    return CustomModel(num_classes=num_classes, extra_dim=extra_dim, pretrained=pretrained, projector_type = projector_type)
