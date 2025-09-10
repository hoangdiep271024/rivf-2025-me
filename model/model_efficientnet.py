import timm
import torch.nn as nn

MODEL_NAME = "efficientnet_b1.ft_in1k"

def build_model(num_classes: int, pretrained: bool = True):
    model_base = timm.create_model(MODEL_NAME, pretrained=pretrained)
    model_base.reset_classifier(num_classes)

    return model_base

