import timm
import torch.nn as nn
import torch

MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"

class CustomModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model_base = timm.create_model(MODEL_NAME, pretrained=pretrained)
        
        # Lấy số chiều embedding
        if hasattr(self.model_base, "classifier"):
            in_features = self.model_base.classifier.in_features
            self.model_base.classifier = nn.Identity()  # bỏ head gốc
        elif hasattr(self.model_base, "fc"):
            in_features = self.model_base.fc.in_features
            self.model_base.fc = nn.Identity()
        else:
            raise ValueError("Không tìm thấy classifier hoặc fc trong model")
        
        # Classifier mới
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, extra_vec=None):
        # Lấy features trước classifier
        features = self.model_base(x)  # shape (B, in_features)
        
        print("Feature shape:", features.shape)  # In ra vector bạn cần
        
        # Nếu có vector bổ sung thì concat
        if extra_vec is not None:
            features = torch.cat([features, extra_vec], dim=1)
        
        # Đưa qua classifier
        out = self.classifier(features)
        return out, features  # trả cả output và feature
        

def build_model(num_classes: int, pretrained: bool = True):
    return CustomModel(num_classes, pretrained)
