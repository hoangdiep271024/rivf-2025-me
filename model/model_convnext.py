# model_convnext.py
from transformers import ConvNextForImageClassification
import torch.nn as nn

MODEL_NAME = "facebook/convnext-large-224-22k-1k"

def build_model(num_classes: int):
    """
    Build a pretrained ConvNeXt model for image classification.
    """
    # Load pretrained ConvNeXt
    model = ConvNextForImageClassification.from_pretrained(MODEL_NAME)
    
    # Reset classifier to match desired num_classes
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model
