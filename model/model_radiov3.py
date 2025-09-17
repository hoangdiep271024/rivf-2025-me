import torch
import torch.nn as nn

MODEL_NAME = "c-radio_v3-b"  

class RadioBackbone(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME, freeze: bool = True):
        super().__init__()
        self.model = torch.hub.load(
            'NVlabs/RADIO', 'radio_model',
            version=model_name, progress=True, skip_validation=True
        )
        self.freeze = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self.hidden_size = None 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                out = self.model(x)
        else:
            out = self.model(x)

        if isinstance(out, (tuple, list)) and len(out) >= 1:
            pooled = out[0]  # (B, C)
        elif isinstance(out, torch.Tensor) and out.dim() == 2:
            pooled = out     # (B, C)
        else:
            raise ValueError("RADIO không trả summary dạng (B, C).")

        if self.hidden_size is None:
            self.hidden_size = pooled.size(1)
        return pooled


class CustomModel(nn.Module):
    def __init__(self, num_classes: int, extra_dim: int = 0,
                 pretrained: bool = True,
                 freeze_backbone: bool = True,
                 model_name: str = MODEL_NAME):
        super().__init__()
        self.model_base = RadioBackbone(model_name=model_name, freeze=freeze_backbone)
        self.extra_dim = extra_dim
        self.classifier = None
        self.extra_proj = None
        self.num_classes = num_classes

    def extract_backbone_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model_base(pixel_values)  # (B, C)

    def _init_layers(self, in_features: int):
        if self.extra_dim > 0:
            self.extra_proj = nn.Sequential(
                nn.Linear(self.extra_dim, in_features),
                nn.BatchNorm1d(in_features),
                nn.ReLU(inplace=True),
            )
            final_in = in_features * 2
        else:
            final_in = in_features
        self.classifier = nn.Linear(final_in, self.num_classes)

    def forward(self, pixel_values: torch.Tensor, extra_vec: torch.Tensor = None,
                return_features: bool = False):
        feat = self.extract_backbone_feature(pixel_values)   # (B, C)
        C = feat.size(1)

        if self.classifier is None:   
            self._init_layers(C)

        if self.extra_proj is not None and extra_vec is not None:
            extra_feat = self.extra_proj(extra_vec)          # (B, C)
            feat = torch.cat([feat, extra_feat], dim=1)      # (B, 2C)

        if return_features:
            return feat
        return self.classifier(feat)


def build_model(num_classes: int, extra_dim: int = 0,
                pretrained: bool = True,
                freeze_backbone: bool = True,
                model_name: str = MODEL_NAME):
    return CustomModel(
        num_classes=num_classes,
        extra_dim=extra_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        model_name=model_name,
    )
