import timm
import torch.nn as nn

def build_model(num_classes: int, pretrained: bool = True):
    model = timm.create_model(
        "convnextv2_tiny",
        pretrained=pretrained,
        num_classes=num_classes,       
    )
    return model

