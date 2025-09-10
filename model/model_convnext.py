import timm
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModeruFC(nn.Module):
    def __init__(self, in_features, num_classes=5, dropout1=0.3, dropout2=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout1)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout2)
        self.out = nn.Linear(128, num_classes)

        self.activation = nn.GELU()  # smoother than ReLU for transformer features

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        x = self.out(x)
        return x


class ConvNeXtPlusModeru(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "convnextv2_atto.fcmae_ft_in1k",
            pretrained=pretrained,
            num_classes=0  # output vector
        )
        self.classifier = ModeruFC(self.backbone.num_features, num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)  # [batch, feature_dim]
        logits = self.classifier(features)
        return logits

def build_model(num_classes=5, pretrained=True, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtPlusModeru(num_classes=num_classes, pretrained=pretrained)
    model.to(device)
    return model
