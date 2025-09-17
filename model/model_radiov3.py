import torch
import torch.nn as nn

MODEL_NAME = "c-radio_v3-b"

class CustomModel(nn.Module):
    def __init__(self, num_classes: int, extra_dim: int = 0, pretrained: bool = True):
        super().__init__()
        # Backbone RADIO
        self.model_base = torch.hub.load(
            'NVlabs/RADIO', 'radio_model',
            version=MODEL_NAME, progress=True, skip_validation=True, pretrained=pretrained
        )
        in_features = 2304  

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

    def forward(self, x, extra_vec=None):
        out = self.model_base(x)
        feat = out[0] if isinstance(out, (tuple, list)) else out  # (B, 2304)

        if self.extra_proj is not None and extra_vec is not None:
            extra_feat = self.extra_proj(extra_vec)  # (B, 2304)
            feat = torch.cat([feat, extra_feat], dim=1)  # (B, 4608)

        return self.classifier(feat)


def build_model(num_classes: int, extra_dim: int = 0, pretrained: bool = True):
    return CustomModel(num_classes=num_classes, extra_dim=extra_dim, pretrained=pretrained)
