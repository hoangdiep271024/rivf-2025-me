import torch
import torch.nn as nn

MODEL_NAME = "c-radio_v3-b"  
class RadioBackbone(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME, freeze: bool = True, summary_dim: int = 768):
        super().__init__()
        self.model = torch.hub.load(
            'NVlabs/RADIO', 'radio_model',
            version=model_name, progress=True, skip_validation=True
        )
        self.hidden_size = summary_dim 
        self.freeze = freeze
        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                out = self.model(pixel_values)
        else:
            out = self.model(pixel_values)
        # RADIO trả (summary, spatial); lấy summary: (B, C)
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            return out[0]
        if isinstance(out, torch.Tensor) and out.dim() == 2:
            return out
        raise ValueError("Không lấy được summary (kỳ vọng outputs[0] có shape (B, C)).")


class CustomModel(nn.Module):
    def __init__(self, num_classes: int, extra_dim: int = 0,
                 pretrained: bool = True,        
                 freeze_backbone: bool = True,
                 model_name: str = MODEL_NAME,
                 summary_dim: int = 768):
        super().__init__()

        # Backbone RADIO (summary)
        self.model_base = RadioBackbone(model_name=model_name, freeze=freeze_backbone, summary_dim=summary_dim)
        in_features = self.model_base.hidden_size

        self.extra_dim = extra_dim
        if extra_dim > 0:
            self.extra_proj = nn.Sequential(
                nn.Linear(extra_dim, in_features),
                nn.BatchNorm1d(in_features),
                nn.ReLU(inplace=True)
            )
            self.in_features = in_features * 2
        else:
            self.extra_proj = None
            self.in_features = in_features

        self.classifier = nn.Linear(self.in_features, num_classes)

    def extract_backbone_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model_base(pixel_values)  # (B, C)

    def forward(self, pixel_values: torch.Tensor, extra_vec: torch.Tensor = None,
                return_features: bool = False):
        feat = self.extract_backbone_feature(pixel_values)  # (B, C)

        if self.extra_proj is not None and extra_vec is not None:
            extra_feat = self.extra_proj(extra_vec)          # (B, C)
            feat = torch.cat([feat, extra_feat], dim=1)      # (B, 2C)

        if return_features:
            return feat

        return self.classifier(feat)


def build_model(num_classes: int, extra_dim: int = 0,
                pretrained: bool = True,
                freeze_backbone: bool = True,
                model_name: str = MODEL_NAME,
                summary_dim: int = 768):
    return CustomModel(
        num_classes=num_classes,
        extra_dim=extra_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        model_name=model_name,
        summary_dim=summary_dim,
    )
