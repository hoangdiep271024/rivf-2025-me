# model_convnext.py
import timm
import torch.nn as nn

MODEL_NAME = "convnextv2_base.fcmae_ft_in1k"

class ConvNextWrapper(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 1024, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(MODEL_NAME, pretrained=pretrained)
        in_features = self.model.get_classifier().in_features
        
        # Thay classifier mạnh hơn: Linear -> ReLU -> Dropout -> Linear
        self.model.reset_classifier(0)  # xóa classifier gốc
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)  # trả về logits trực tiếp

def build_model(num_classes: int):
    return ConvNextWrapper(num_classes)
