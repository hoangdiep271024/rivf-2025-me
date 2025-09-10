import timm
import torch
import torch.nn as nn

class ModeruFC(nn.Module):
    def __init__(self, in_features, num_classes=5):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
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
