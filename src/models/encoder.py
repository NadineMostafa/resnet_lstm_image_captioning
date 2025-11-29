import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.CNNBackbone = nn.Sequential(
            *list(self.backbone.children())[:-2]
        )
        self.fc = nn.Linear(self.backbone.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.CNNBackbone(images)

        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        features = self.bn(self.fc(features))

        return features