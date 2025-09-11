# model_convnext.py
import timm
import torch.nn as nn

MODEL_NAME = "convnextv2_base.fcmae_ft_in1k"

def build_model(num_classes: int):
    # Load pretrained ConvNeXtV2
    model = timm.create_model(MODEL_NAME, pretrained=True)
    
    # Reset classifier về đúng số lớp, 1 Linear cuối cùng
    model.reset_classifier(num_classes)
    
    return model
