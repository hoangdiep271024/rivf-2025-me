import timm
import torch
import torch.nn as nn

MODEL_NAME = "densenet121.ra_in1k"

class CustomDenseNet(nn.Module):
    def __init__(self, model_name: str, num_classes: int, extra_dim: int = 0, pretrained: bool = True):
        super().__init__()
        self.model_base = timm.create_model(model_name, pretrained=pretrained)

        if hasattr(self.model_base, "classifier"):
            in_features = self.model_base.classifier.in_features
            self.model_base.classifier = nn.Identity()
        else:
            raise ValueError("Không tìm thấy classifier trong DenseNet")

        self.backbone_dim = in_features
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
        features = self.model_base(x)  # (B, backbone_dim)

        if self.extra_proj is not None and extra_vec is not None:
            if extra_vec.dim() == 1:
                extra_vec = extra_vec.unsqueeze(0)
            if features.size(0) != extra_vec.size(0):
                raise ValueError(
                    f"Batch size không khớp: features={features.size()}, extra_vec={extra_vec.size()}"
                )

            extra_feat = self.extra_proj(extra_vec) 
            features = torch.cat([features, extra_feat], dim=1)  # (B, backbone_dim*2)

        out = self.classifier(features)
        return out


def build_model(num_classes: int, extra_dim: int = 0, pretrained: bool = True, model_name: str = MODEL_NAME):
    return CustomDenseNet(
        model_name=model_name,
        num_classes=num_classes,
        extra_dim=extra_dim,
        pretrained=pretrained
    )

