import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import argparse
from torchvision.ops import nms
from transformers import OwlViTForObjectDetection, OwlViTProcessor

class BoundingBoxExtractor:
    """Extract bounding boxes from images."""
    
    def __init__(self, device='cuda:4' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model, self.processor = self.load_object_detection_model()

    def load_object_detection_model(self):
        processor = OwlViTProcessor.from_pretrained('google/owlvit-large-patch14')
        model = OwlViTForObjectDetection.from_pretrained('google/owlvit-large-patch14')
        model.eval()
        model.to(self.device)

        return model, processor

    def load_image(self, image_path):
        image = Image.open(image_path)
        image = image.convert('RGB')
        return image

    def detect_objects_in_image(self, image, texts=[['object', 'can', 'bottle']]) -> dict:
        inputs = self.processor(text=texts, images=image, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**inputs.to(self.device))

        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]['boxes'], results[i]['scores'], results[i]['labels']

        return boxes, scores, labels

    def is_box_overlapping_image(self, image, box, threshold=0.5) -> bool:
        img_width, img_height = image.size
        image_area = img_width * img_height
        x0, y0, x1, y1 = box
        box_area = (x1 - x0) * (y1 - y0)
        overlap_ratio = box_area / image_area

        if overlap_ratio > threshold:
            return True
        else:
            return False

    def apply_nms(self, image, boxes, scores, iou_threshold=0.5, score_threshold=0.1) -> torch.Tensor:
        boxes_filtered = boxes[scores > score_threshold]
        scores_filtered = scores[scores > score_threshold]

        nms_out = nms(boxes_filtered, scores_filtered, iou_threshold=iou_threshold)

        bounding_boxes = boxes_filtered[nms_out]
        boxes_final = []
        for i in range(len(bounding_boxes)):
            if not self.is_box_overlapping_image(image, bounding_boxes[i], threshold=0.5):
                boxes_final.append(bounding_boxes[i])

        return torch.stack(boxes_final)

    def crop_boxes(self, image, boxes) -> list:
        cropped_boxes = []
        for (x_min, y_min, x_max, y_max) in boxes:
            cropped_boxes.append(image.crop((int(x_min), int(y_min), int(x_max), int(y_max))))

        return cropped_boxes

    def extract_bounding_boxes(self, image_path, output_path, save=False) -> list:
        image = self.load_image(image_path)
        boxes, scores, labels = self.detect_objects_in_image(image)
        boxes = self.apply_nms(image, boxes, scores)
        cropped_boxes = self.crop_boxes(image, boxes)

        if save:
            for i, cropped_box in enumerate(cropped_boxes):
                cropped_box.save(f'{output_path}/{i}.jpg')

        return cropped_boxes

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', type=str, default='cuda:4' if torch.cuda.is_available() else 'cpu')
    argparser.add_argument('--image_path', type=str, default='/data/mburges/for_matthias/images_hackathon/image0.jpg')
    argparser.add_argument('--output_path', type=str, default='data/images/')
    args = argparser.parse_args()

    extractor = BoundingBoxExtractor(device=args.device)
    extractor.extract_bounding_boxes(args.image_path, args.output_path, save=True)
