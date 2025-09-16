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

        # tổng số chiều sau khi concat
        self.backbone_dim = in_features
        self.extra_dim = extra_dim
        self.in_features = in_features + extra_dim

        # classifier cuối
        self.classifier = nn.Linear(self.in_features, num_classes)

        print(f"[Init] Backbone dim: {self.backbone_dim}, Extra dim: {self.extra_dim}, "
              f"Classifier input: {self.in_features}")

    def forward(self, x, extra_vec=None):
        features = self.model_base(x)  # (B, backbone_dim)

        if extra_vec is not None:
            if extra_vec.dim() == 1:  # nếu chỉ có (extra_dim,) thì thêm batch
                extra_vec = extra_vec.unsqueeze(0)
            # đảm bảo cùng batch size
            if features.size(0) != extra_vec.size(0):
                raise ValueError(f"Batch size không khớp: features={features.size()}, extra_vec={extra_vec.size()}")
            features = torch.cat([features, extra_vec], dim=1)  # (B, backbone+extra)

        out = self.classifier(features)
        return out, features

def build_model(num_classes: int, extra_dim: int = 0, pretrained: bool = True, model_name: str = "tf_efficientnetv2_s.in21k_ft_in1k"):
    return CustomModel(model_name=model_name, 
                       num_classes=num_classes, 
                       extra_dim=extra_dim, 
                       pretrained=pretrained)

