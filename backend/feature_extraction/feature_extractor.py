

import timm
import torch
import numpy as np
import hdbscan
from torch import nn
from sklearn import decomposition


class FeatureExtractor(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m
        self.pca = decomposition.PCA(n_components=2)
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m.forward_features(x)

    @torch.inference_mode()
    def forward_project_pca(self, x: torch.Tensor) -> np.ndarray:
        z = self.m(x).numpy()
        self.pca.fit(z)
        y = self.pca.transform(z)
        return y


# PCR, umap
def get_feature_extractor(name: str = 'convnext_base.clip_laion2b') -> FeatureExtractor:
    m = timm.create_model(model_name=name, pretrained=True)
    return FeatureExtractor(m)


