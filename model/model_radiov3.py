import torch
import torch.nn as nn
import math

class ModeruCNN(nn.Module):
    def __init__(self, in_channels=768, num_classes=5, patch_grid: tuple = None):
        super().__init__()
        self.patch_grid = patch_grid

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        x: patch features từ RADIO, shape (B, N, D)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  

        B, N, D = x.shape
        if self.patch_grid is not None:
            h, w = self.patch_grid
            assert h * w == N, f"patch_grid {self.patch_grid} không khớp với N={N}"
        else:
            h = int(math.sqrt(N))
            while N % h != 0:
                h -= 1
            w = N // h

        x = x.transpose(1, 2).reshape(B, D, h, w)

        # CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class RadioWithCNN(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = True, patch_grid: tuple = None):
        super().__init__()
        self.backbone = torch.hub.load(
            'NVlabs/RADIO',
            'radio_model',
            version="c-radio_v3-b",
            progress=True,
            skip_validation=True
        )

        hidden_size = 768 
        self.patch_grid = patch_grid

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.cnn_classifier = ModeruCNN(
            in_channels=hidden_size,
            num_classes=num_classes,
            patch_grid=patch_grid
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        if hasattr(outputs, "__getitem__") and len(outputs) > 1:
            patch_features = outputs[1]  
        else:
            raise ValueError("Không tìm thấy patch features từ RADIO output")

        logits = self.cnn_classifier(patch_features)
        return logits


def build_model(num_classes: int, freeze_backbone: bool = True, patch_grid: tuple = None):
    return RadioWithCNN(num_classes=num_classes, freeze_backbone=freeze_backbone, patch_grid=patch_grid)


