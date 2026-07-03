import timm
import torch.nn as nn

class EfficientNetB0(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0
        )
        self.classifier = nn.Sequential(
            nn.Linear(1280, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.model(x)
        return self.classifier(features)
