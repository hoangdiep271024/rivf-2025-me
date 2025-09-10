import torch
import torch.nn as nn
from transformers import AutoModel


# CNN classifier của bạn
class Moderu_cnn(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # bỏ conv5, thay bằng AdaptiveAvgPool2d để giữ kích thước ổn định
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = self.conv3(x) 
        x = self.conv4(x) 
        x = self.avgpool(x)  
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x



# Ghép DINOv3 backbone + CNN classifier
class DinoWithCNN(nn.Module):
    def __init__(self, num_classes: int, pretrained_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        hidden_size = self.backbone.config.hidden_size  # thường = 768
        assert hidden_size == 768, f"Hiện code chỉ support hidden_size=768, nhưng model backbone trả về {hidden_size}"
        self.reshape_dims = (3, 16, 16)
        assert torch.prod(torch.tensor(self.reshape_dims)) == hidden_size

        # CNN classifier
        self.cnn_classifier = Moderu_cnn(in_channels=3, num_classes=num_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_feature = outputs.last_hidden_state[:, 0]  # (B, 768)

        # reshape -> (B, 3, 16, 16)
        x = cls_feature.view(cls_feature.size(0), *self.reshape_dims)

        logits = self.cnn_classifier(x)
        return logits


def build_model(num_classes: int):
    return DinoWithCNN(num_classes=num_classes)
