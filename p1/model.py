from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn as nn

def get_model(num_classes=4, pretrained=True):
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_f = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_f, num_classes)
    )
    return model
