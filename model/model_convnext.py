# model_convnext.py
import timm
import torch.nn as nn

MODEL_NAME = "vit_small_patch16_224.augreg_in21k_ft_in1k"

def build_model(num_classes: int, hidden_dim: int = 512, dropout: float = 0.3):
    model = timm.create_model(MODEL_NAME, pretrained=True)
    in_features = model.get_classifier().in_features
    
    # Thay classifier mạnh hơn: Linear -> ReLU -> Dropout -> Linear
    model.reset_classifier(0)
    model.classifier = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes)
    )
    
    return model
