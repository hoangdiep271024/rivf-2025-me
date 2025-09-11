# model_convnext.py
from transformers import ConvNextForImageClassification
import torch.nn as nn

MODEL_NAME = "facebook/convnext-large-224-22k-1k"

class ConvNextWrapper(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = ConvNextForImageClassification.from_pretrained(MODEL_NAME)
        # Reset classifier
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        # Trả về logits trực tiếp để train_one_epoch không cần sửa
        return self.model(x).logits

def build_model(num_classes: int):
    return ConvNextWrapper(num_classes)
