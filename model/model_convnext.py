import timm

MODEL_NAME = "vit_small_patch16_224.augreg_in21k_ft_in1k"

def build_model(num_classes: int):
    model = timm.create_model(MODEL_NAME, pretrained=True)
    
    # Chỉ dùng 1 Linear cuối cùng
    model.reset_classifier(num_classes)
    
    return model
