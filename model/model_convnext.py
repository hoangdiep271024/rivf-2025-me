# model.py
from transformers import ConvNextForImageClassification
import torch.nn as nn

MODEL_NAME = "facebook/convnext-large-224-22k-1k"

def build_model(num_classes: int):
    """
    Build a pretrained ConvNeXt model for image classification.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        model: ConvNeXtForImageClassification instance
    """
    # Load pretrained ConvNeXt
    model = ConvNextForImageClassification.from_pretrained(MODEL_NAME)
    
    # Reset classifier to match desired num_classes
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    return model
