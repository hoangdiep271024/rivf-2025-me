import timm
import torch.nn as nn

def build_model(num_classes: int, pretrained: bool = True):
    model = timm.create_model(
        "vgg13_bn",
        pretrained=pretrained,
        num_classes=num_classes,       
    )
    return model