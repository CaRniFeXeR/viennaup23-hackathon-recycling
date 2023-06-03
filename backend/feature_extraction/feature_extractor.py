import glob
import os
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import timm
import torch
import numpy as np
import hdbscan
from torch import nn
from sklearn import decomposition
from torchvision import transforms

def mkdir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


class FeatureExtractor(nn.Module):
    """
        Composable Feature Extractor
    """
    def __init__(self, img_path: Optional[Union[str, os.PathLike]], extractor: nn.Module, device: str = 'cpu'):
        """

        :param img_path: Directory of Images or Path to Image. Loads entire batch into memory if passed
        :param extractor: torch module with pretrained weights to extract features from image datg
        :param device:
        """
        super().__init__()
        self.pca = decomposition.PCA(n_components=2)
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True)
        self.device = device
        self.resize = transforms.Resize(size=(256, 256), interpolation=transforms.InterpolationMode.NEAREST)
        if img_path:
            self.load_imgs(img_path)
        else:
            self.imgs = None
        self.extractor = extractor.to(device)

    def load_imgs(self, img_path: Union[str, os.PathLike]):
        if os.path.isdir(img_path):
            files = glob.glob(f'{img_path}/*.*')
            arr = np.array([np.array(self.resize(Image.open(fname))) for fname in files])
        else:
            arr = np.array([np.array(self.resize(Image.open(img_path)))])
        self.imgs = torch.tensor(np.transpose(arr, (0, 3, 1, 2)))

    def _extract_features_preloaded(self) -> torch.tensor:
        return self.extractor.forward_features(self.imgs)

    def _extract_features(self, x: torch.Tensor) -> torch.tensor:
        x = x.to(self.device)
        return self.extractor.forward_features(x)

    @torch.inference_mode()
    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param x: Optional image tensor to extract features from. If None, extracts feature from preloaded images
        :return:
        """
        if x is None:
            assert self.imgs, "Preload images or pass torch tensor"
            return self.m._extract_features_preloaded(x)
        return self._extract_features(x)

    @torch.inference_mode()
    def forward_project_pca(self, x: torch.Tensor) -> np.ndarray:
        z = self.m(x).numpy()
        self.pca.fit(z)
        y = self.pca.transform(z)
        return y


def get_feature_extractor(img_path: Optional[Union[str, os.PathLike]] = None,
                          device: Optional[str] = None,
                          extractor_name: str = 'convnext_base.clip_laion2b') -> FeatureExtractor:
    extractor = timm.create_model(model_name=extractor_name, pretrained=True)
    device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
    return FeatureExtractor(img_path=img_path,
                            extractor= extractor,
                            device=device)
