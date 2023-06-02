import timm
import torch
from torch import nn

class FeatureExtractor(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m.forward_features(x)


def get_feature_extractor(name: str = 'convnext_base.clip_laion2b') -> FeatureExtractor:
    m = timm.create_model(name=name, pretrained=True)
    return FeatureExtractor(m)
