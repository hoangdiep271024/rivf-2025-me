import timm
import torch.nn as nn

MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"

def build_model(num_classes: int, pretrained: bool = True):
    """
    Build model từ timm với model name cố định.
    """
    model_base = timm.create_model(MODEL_NAME, pretrained=pretrained)

    # Thay head cuối bằng num_classes
    if hasattr(model_base, "classifier"):  # EfficientNet, MobileNet...
        in_features = model_base.classifier.in_features
        model_base.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model_base, "fc"):  # ResNet...
        in_features = model_base.fc.in_features
        model_base.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Không tìm thấy head cho {MODEL_NAME}")

    return model_base
