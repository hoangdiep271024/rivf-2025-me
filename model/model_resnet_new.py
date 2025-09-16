import timm
import torch.nn as nn
import torch

MODEL_NAME = "resnet50.a1_in1k"

class CustomModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, extra_dim: int = 0, pretrained: bool = True):
        super().__init__()
        # backbone
        self.model_base = timm.create_model(model_name, pretrained=pretrained)

        # lấy số chiều feature gốc
        if hasattr(self.model_base, "classifier"):
            in_features = self.model_base.classifier.in_features
            self.model_base.classifier = nn.Identity()
        elif hasattr(self.model_base, "fc"):
            in_features = self.model_base.fc.in_features
            self.model_base.fc = nn.Identity()
        else:
            raise ValueError("Không tìm thấy classifier hoặc fc trong model")

        self.backbone_dim = in_features
        self.extra_dim = extra_dim
        self.in_features = in_features + extra_dim

        # classifier cuối
        self.classifier = nn.Linear(self.in_features, num_classes)

        print(f"[Init] Backbone dim: {self.backbone_dim}, Extra dim: {self.extra_dim}, "
              f"Classifier input: {self.in_features}")

    def forward(self, x, extra_vec=None):
        features = self.model_base(x)  # (B, backbone_dim)

        print(f"Backbone feature: min={features.min().item():.4f}, "
          f"max={features.max().item():.4f}, "
          f"mean={features.mean().item():.4f}")

        if extra_vec is not None:
            print(f"Extra vec: min={extra_vec.min().item():.4f}, "
              f"max={extra_vec.max().item():.4f}, "
              f"mean={extra_vec.mean().item():.4f}")

            features = torch.cat([features, extra_vec], dim=1)

            print(f"Concat feature: min={features.min().item():.4f}, "
              f"max={features.max().item():.4f}, "
              f"mean={features.mean().item():.4f}")

        out = self.classifier(features)
        return out, features
  

def build_model(num_classes: int, extra_dim: int = 0, pretrained: bool = True, model_name: str = "tf_efficientnetv2_s.in21k_ft_in1k"):
    return CustomModel(model_name=model_name, 
                       num_classes=num_classes, 
                       extra_dim=extra_dim, 
                       pretrained=pretrained)

