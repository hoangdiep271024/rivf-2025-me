# model.py
import torch.nn as nn
from transformers import AutoModel

class SigLIPClassifier(nn.Module):
    def __init__(self, num_classes: int, ckpt: str = "google/siglip2-base-patch16-224", device_map="auto"):
        super().__init__()
        # Backbone SigLIP
        self.backbone = AutoModel.from_pretrained(ckpt, device_map=device_map)
        
        # Lấy hidden size của backbone (thường là 768)
        hidden_size = self.backbone.config.hidden_size
        
        # Classifier head
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        # Lấy output từ backbone
        outputs = self.backbone(pixel_values=pixel_values)
        
        # Lấy CLS token (vị trí 0)
        pooled = outputs.last_hidden_state[:, 0]
        
        # Đưa qua classifier
        return self.classifier(pooled)


def build_model(num_classes: int, pretrained: bool = True):
    model = SigLIPClassifier(num_classes=num_classes)
    return model
