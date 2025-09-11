# model_convnext.py
from transformers import ConvNextForImageClassification
import torch.nn as nn

MODEL_NAME = "convnextv2_base.fcmae_ft_in22k_in1k"

class ConvNextWrapper(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.model = ConvNextForImageClassification.from_pretrained(MODEL_NAME)
        in_features = self.model.classifier.in_features
        
        # Mạnh hơn: thêm 1 hidden layer với ReLU + Dropout
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Trả về logits trực tiếp
        return self.model(x)

def build_model(num_classes: int):
    return ConvNextWrapper(num_classes)
