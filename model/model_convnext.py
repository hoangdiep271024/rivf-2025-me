# model_convnext.py
import timm
import torch.nn as nn

MODEL_NAME = "vit_small_patch16_224.augreg_in21k_ft_in1k"

class ViTWrapper(nn.Module):
    def __init__(self, num_classes=5, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.model = timm.create_model(MODEL_NAME, pretrained=True)
        in_features = self.model.get_classifier().in_features
        
        # Linear -> ReLU -> Dropout -> Linear, vẫn trả về size=num_classes
        self.model.reset_classifier(0)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)  # trả về logits [batch, num_classes]

def build_model(num_classes=5):
    return ViTWrapper(num_classes)
