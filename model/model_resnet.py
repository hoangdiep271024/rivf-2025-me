import timm
import torch
import torch.nn as nn

MODEL_NAME = "resnet50.a1_in1k"

class CustomModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model_base = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0)
        in_features = self.model_base.num_features
        self.backbone_dim = in_features

        self.norm = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.model_base(x)  
        features = self.norm(features) 
        out = self.classifier(features)
        return out

def build_model(num_classes: int, pretrained: bool = True):
    return CustomModel(num_classes=num_classes, pretrained=pretrained)