# model.py
import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification

class SigLIPClassifier(nn.Module):
    def __init__(self, num_classes: int, ckpt: str = "google/siglip-base-patch16-224"):
        super().__init__()
        # Load model pretrained
        self.model = AutoModelForImageClassification.from_pretrained(ckpt)
        
        # Lấy số features của classifier hiện tại
        in_features = self.model.classifier.in_features
        
        # Thay classifier head nếu số class khác pretrained
        if num_classes != self.model.config.num_labels:
            self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, pixel_values):
        """
        pixel_values: Tensor [B, C, H, W], đã chuẩn hóa
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

def build_model(num_classes: int, ckpt: str = "google/siglip-base-patch16-224"):
    return SigLIPClassifier(num_classes=num_classes, ckpt=ckpt)
