import glob
import os
import random
from pathlib import Path
from typing import List, Optional, Union
import PIL.Image as PILImage
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

    def __init__(self,
                 img_path: Optional[Union[str, os.PathLike]],
                 extractor: nn.Module, device: str = 'cpu'):
        """

        :param img_path: Directory of Images or Path to Image. Loads entire batch into memory if passed
        :param extractor: torch module with pretrained weights to extract features from image datg
        :param device:
        """
        super().__init__()
        self.extractor = extractor.to(device)
        self.pca = decomposition.PCA(n_components=64)
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True)
        self.device = device
        self.resize = transforms.Resize(size=(256, 256), interpolation=transforms.InterpolationMode.NEAREST)
        self.to_tensor = transforms.ToTensor()
        self.transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=(30, 210), translate=(0.1, 0.3), scale=(0.5, 0.75))
            # Add more transformations to the set if desired
        ]
        if img_path:
            self.load_imgs(img_path)
            self._fit_pca()
        else:
            self.imgs = None

    def load_imgs(self, img_path: Union[str, os.PathLike]):
        if os.path.isdir(img_path):
            files = glob.glob(f'{img_path}/*.*')
            self.imgs = [self.to_tensor(Image.open(fname)).unsqueeze(dim=0) for fname in files]
            # arr = np.array([np.array(self.resize(Image.open(fname))) for fname in files])
        else:
            # arr = np.array([np.array(self.resize(Image.open(img_path)))])
            self.imgs = [self.to_tensor(Image.open(img_path)).unsqueeze(dim=0)]

        # self.imgs = torch.tensor(np.transpose(arr, (0, 3, 1, 2)))

    def _fit_pca(self, imgs: Optional[torch.Tensor] = None):
        if imgs:
            imgs = imgs.numpy()
            self.pca.fit(imgs)
        else:
            f = self._extract_features_preloaded()
            self.pca.fit(f)

    def _extract_features_preloaded(self) -> torch.tensor:
        features = []
        for i in self.imgs:
            features.append(self.extractor.head.global_pool(self.extractor.forward_features(i)))
        features = torch.cat(features, dim=0).squeeze()
        if len(features.shape) == 1:
            features = features.unsqueeze(dim=0)
        return features

    def extract_feature(self, img: Union[Image.Image, torch.Tensor], augment: bool = True) -> torch.Tensor:
        img = self.to_tensor(img)
        if augment:
            for transform in self.transform_list:
                if random.random() < 0.5:
                    img = transform(img)
        if len(img.shape) == 3:
            img = img.unsqueeze(dim=0)
        return self.extractor(img)

        # accept pil image + select a random affine transformation

    def _extract_features(self, x: torch.Tensor) -> torch.tensor:
        x = x.to(self.device)
        return self.extractor.head.global_pool(self.extractor.forward_features(x))

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract(x)

    @torch.inference_mode()
    def extract(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param x: Optional image tensor to extract features from. If None, extracts feature from preloaded images
        :return:
        """
        if x is None:
            assert self.imgs, "Preload images or pass torch tensor"
            return self._extract_features_preloaded()
        return self._extract_features(x)

    @torch.inference_mode()
    def pca_project(self, x: torch.Tensor) -> np.ndarray:
        y = self.pca.transform(x)
        return y


def get_feature_extractor(img_path: Optional[Union[str, os.PathLike]] = None,
                          device: Optional[str] = None,
                          extractor_name: str = 'convnext_small.fb_in22k') -> FeatureExtractor:
    extractor = timm.create_model(model_name=extractor_name, pretrained=True, num_classes=0)
    device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
    return FeatureExtractor(img_path=img_path,
                            extractor=extractor,
                            device=device)


if __name__ == '__main__':
    # one npy file per feature
    img_path = os.environ.get('INPUT_PATH',
                              '/home/alireza/repositories/viennaup23-hackathon-recycling/backend/data/images')
    feature_path = Path(os.environ.get('OUTPUT_PATH',
                                       '/home/alireza/repositories/viennaup23-hackathon-recycling/backend/data/features_noclip64'))
    pca_proj = bool(os.environ.get('PCA_PROJ', False))
    mkdir(feature_path)
    file_list = glob.glob(f'{img_path}/*.*')
    file_list.sort()
    m = get_feature_extractor(img_path=img_path)
    features = m.extract()
    if pca_proj:
        features = m.pca_project(features)
    for idx, feature_vec in enumerate(features):
        np.save(feature_path / Path(file_list[idx]).stem, feature_vec)
