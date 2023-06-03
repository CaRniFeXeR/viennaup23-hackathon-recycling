import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision.transforms import ToTensor
from region_proposal.object_detection import BoundingBoxExtractor
from feature_extraction.feature_extractor import FeatureExtractor, get_feature_extractor

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', type=str, default='cuda:4' if torch.cuda.is_available() else 'cpu')
    argparser.add_argument('--image_path', type=str, default='/data/mburges/for_matthias/images_hackathon/image2.jpg')
    argparser.add_argument('--output_path', type=str, default='_output/')
    args = argparser.parse_args()

    device = torch.device(args.device)

    # check that output_path does not exist and create it
    assert not Path(args.output_path).exists(), 'Output path already exists'
    Path(args.output_path).mkdir(parents=True, exist_ok=False)

    print('## Extracting bounding boxes ##')
    object_detector = BoundingBoxExtractor(device)
    boxes = object_detector.extract_bounding_boxes(args.image_path, args.output_path, save=True)

    print('## Extracting features ##')
    feature_extractor = get_feature_extractor(device=device)
    feature_extractor = feature_extractor.to(device)

    to_tensor = ToTensor()
    for i, box in enumerate(boxes):
        box_t = to_tensor(box).unsqueeze(0).to(device)
        features = feature_extractor(box_t).cpu().detach().numpy()[:, :, 0, 0]


    print('test')