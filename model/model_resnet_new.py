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
        features = self.model_base(x) 

        if extra_vec is not None and extra_proj í not None:
            if extra_vec.dim() == 1:  
                extra_vec = extra_vec.unsqueeze(0)
            if features.size(0) != extra_vec.size(0):
                raise ValueError(f"Batch size không khớp: features={features.size()}, extra_vec={extra_vec.size()}")
            
            extra_feat = self.extra_proj(extra_vec)  
            features = torch.cat([features, extra_feat], dim=1) 

        out = self.classifier(features)
        return out
  

def build_model(num_classes: int, extra_dim: int = 0, pretrained: bool = True, model_name: str = MODEL_NAME):
    return CustomModel(model_name=model_name, 
                       num_classes=num_classes, 
                       extra_dim=extra_dim, 
                       pretrained=pretrained)
