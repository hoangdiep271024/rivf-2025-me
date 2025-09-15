import timm
import torch.nn as nn
import torch

MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"

class CustomModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, extra_dim: int = 0):
        super().__init__()
        self.model_base = timm.create_model(MODEL_NAME, pretrained=pretrained)

        if hasattr(self.model_base, "classifier"):
            in_features = self.model_base.classifier.in_features
            self.model_base.classifier = nn.Identity()
        elif hasattr(self.model_base, "fc"):
            in_features = self.model_base.fc.in_features
            self.model_base.fc = nn.Identity()
        else:
            raise ValueError("Không tìm thấy classifier hoặc fc trong model")

        # Nếu có vector phụ thì cộng thêm dimension
        self.in_features = in_features + extra_dim
        self.classifier = nn.Linear(self.in_features, num_classes)

    def forward(self, x, extra_vec=None, return_features=False):
        features = self.model_base(x)  # (B, in_features)

        if extra_vec is not None:
            features = torch.cat([features, extra_vec], dim=1)

        out = self.classifier(features)

        if return_features:
            return out, features
        return out

        

def build_model(num_classes: int, pretrained: bool = True):
    return CustomModel(num_classes, pretrained)
