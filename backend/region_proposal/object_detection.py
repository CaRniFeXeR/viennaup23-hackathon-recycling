import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import argparse
from torchvision.ops import nms
from transformers import OwlViTForObjectDetection, OwlViTProcessor

def load_object_detection_model(device: torch.device) -> nn.Module:
    processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")
    model.eval()
    model.to(device)

    return model, processor

def load_image(image_path: Path) -> Image:
    image = Image.open(image_path)
    image = image.convert("RGB")
    return image

def detect_objects_in_image(
    model: OwlViTForObjectDetection,
    processor: OwlViTProcessor,
    image: Image,
    device: torch.device,
    texts: list = [["object", "can", "bottle"]]
) -> dict:
    inputs = processor(text=texts, images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs.to(device))

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]]).to(device)
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    return boxes, scores, labels

def is_box_overlapping_image(img: Image, box, threshold: float = 0.5) -> bool:
    """Check if the bounding box overlaps with the image."""
    # Get the image dimensions
    img_width, img_height = img.size
    
    # Calculate the area of the image
    image_area = img_width * img_height
    
    # Extract the coordinates of the bounding box
    x0, y0, x1, y1 = box
    
    # Calculate the area of the bounding box
    box_area = (x1 - x0) * (y1 - y0)
    
    # Calculate the overlap ratio
    overlap_ratio = box_area / image_area
    
    # Check if the overlap ratio is larger than the threshold
    if overlap_ratio > threshold:
        return True
    else:
        return False

def apply_nms(
    image: Image,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.1
) -> torch.Tensor:
    """Apply non-maximum suppression at test time to avoid detecting too many objects."""
    boxes_filtered = boxes[scores > score_threshold]
    scores_filtered = scores[scores > score_threshold]

    nms_out = nms(boxes_filtered, scores_filtered, iou_threshold=iou_threshold)

    bounding_boxes = boxes_filtered[nms_out]
    boxes_final = []
    for i in range(len(bounding_boxes)):
        if not is_box_overlapping_image(image, bounding_boxes[i], threshold=0.5):
            boxes_final.append(bounding_boxes[i])

    return torch.stack(boxes_final)

def crop_boxes(image, boxes):
    cropped_boxes = []
    for (x_min, y_min, x_max, y_max) in boxes:
        cropped_boxes.append(image.crop((int(x_min), int(y_min), int(x_max), int(y_max))))
        
    return cropped_boxes

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--device", type=str, default="cuda:4" if torch.cuda.is_available() else "cpu")
    argparser.add_argument("--image_path", type=str, default="/data/mburges/for_matthias/images_hackathon/image0.jpg")
    argparser.add_argument("--output_path", type=str, default="data/images/")
    args = argparser.parse_args()

    # load model
    device = torch.device(args.device)
    model, processor = load_object_detection_model(device)

    #load image
    print('## Load model')
    image = load_image(args.image_path)

    # detect objects in image
    print('## Detect objects in image')
    boxes, scores, labels = detect_objects_in_image(model, processor, image, device)

    # apply non-maximum suppression and crop boxes
    print('## Apply non-maximum suppression and crop boxes')
    boxes = apply_nms(image, boxes, scores)
    cropped_boxes = crop_boxes(image, boxes)

    # save cropped boxes
    print('## Save cropped boxes')
    for i, cropped_box in enumerate(cropped_boxes):
        cropped_box.save(f"data/images/{i}.jpg")