import torch
import torch.nn as nn

MODEL_NAME = "c-radio_v3-b"

class CustomModel(nn.Module):
    def __init__(self, num_classes: int, extra_dim: int = 0, pretrained: bool = True):
        super().__init__()
        # Backbone RADIO
        if pretrained:
            self.model_base = torch.hub.load(
                'NVlabs/RADIO', 'radio_model',
                version=MODEL_NAME, progress=True, skip_validation=True
            )
        else:
            # không load pretrained (random init)
            self.model_base = torch.hub.load(
                'NVlabs/RADIO', 'radio_model',
                version=MODEL_NAME, progress=False, skip_validation=True,
                pretrained=False
            )

        self.extra_dim = extra_dim
        self.classifier = None
        self.extra_proj = None
        self.num_classes = num_classes

    def forward(self, x, extra_vec=None):
        out = self.model_base(x)
        feat = out[0] if isinstance(out, (tuple, list)) else out  # (B, C)
        C = feat.size(1)

        if self.classifier is None:
            device, dtype = feat.device, feat.dtype
            if self.extra_dim > 0:
                self.extra_proj = nn.Sequential(
                    nn.Linear(self.extra_dim, C),
                    nn.BatchNorm1d(C),
                    nn.ReLU(inplace=True)
                ).to(device=device, dtype=dtype)
                in_features = 2 * C
            else:
                in_features = C
            self.classifier = nn.Linear(in_features, self.num_classes).to(device=device, dtype=dtype)

        if self.extra_proj is not None and extra_vec is not None:
            extra_vec = extra_vec.to(device=feat.device, dtype=feat.dtype)
            extra_feat = self.extra_proj(extra_vec)
            feat = torch.cat([feat, extra_feat], dim=1)

        return self.classifier(feat)


def build_model(num_classes: int, extra_dim: int = 0, pretrained: bool = True):
    """
    Build RADIO model + optional extra vector.
    """
    return CustomModel(num_classes=num_classes, extra_dim=extra_dim, pretrained=pretrained)
