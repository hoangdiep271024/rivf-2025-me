# model.py
import torch
import torch.nn as nn
from transformers import AutoModel

class DinoV3Classifier(nn.Module):
    def __init__(self, num_classes: int, pretrained_model_name: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_feature = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_feature)
        return logits

def build_model(num_classes: int):
    return DinoV3Classifier(num_classes=num_classes)
