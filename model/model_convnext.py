# model_convnext.py
import timm
import torch.nn as nn

MODEL_NAME = "vit_base_patch16_224"

def build_model(num_classes: int):
    # Load pretrained ConvNeXtV2
    model = timm.create_model(MODEL_NAME, pretrained=True)
    
    # Reset classifier về đúng số lớp, 1 Linear cuối cùng
    model.reset_classifier(num_classes)
    
    return model
