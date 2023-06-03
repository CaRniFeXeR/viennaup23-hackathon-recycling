import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw
from torchinfo import summary
from torchvision.ops import nms
from tqdm.auto import tqdm
from transformers import OwlViTForObjectDetection, OwlViTProcessor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=125):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def get_features():

    processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")
    model.eval;
    model.to("cuda");

    feature_embeddings = []

    # print all named model parameters
    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    n = 0

    images = [img for img in Path('/data/mburges/for_matthias/images_hackathon').iterdir()]
    for image_path in tqdm(images):
        # Load image
        image = Image.open(image_path).convert("RGB")
        #texts = [["object", "can", "bottle"]]
        texts = [["object", "Can: Sealed cylindrical metal container", "Bottle: Narrow-necked liquid storage container"]]
        inputs = processor(text=texts, images=image, return_tensors="pt")

        n += 1

        with torch.no_grad():
            outputs = model(**inputs.to("cuda"))

        # print(outputs.keys())
        # for k, v in outputs.items():
        #     print(k, v.shape)
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]]).to("cuda")
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        # # Print detected objects and rescaled box coordinates
        # score_threshold = 0.1
        # for box, score, label in zip(boxes, scores, labels):
        #     box = [round(i, 2) for i in box.tolist()]
        #     if score >= score_threshold:
        #         print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

        boxes_filtered = boxes[scores > 0.1]
        scores_filtered = scores[scores > 0.1]
        labels_filtered = labels[scores > 0.1]
        embeddings_filtered = outputs['class_embeds'][0, scores > 0.1]

        # print("boxes", boxes.shape)

        # for k, val in outputs.items():
        #     if k not in {"text_model_output", "vision_model_output"}:
        #         print(f"{k}: shape of {val.shape}")

        nms_out = nms(boxes_filtered, scores_filtered, iou_threshold=0.1)

        image_path = images[n-1]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # visualize results with pillow
        draw = ImageDraw.Draw(image)
        bounding_boxes = boxes_filtered[nms_out]
        embeddings_nms = embeddings_filtered[nms_out]
        labels_nms = labels_filtered[nms_out]

        labels_in_image = []
        for i in labels_nms:
            labels_in_image.append(text[i])

        # for i in range(len(bounding_boxes)):
        #     # label
        #     draw.text((bounding_boxes[i][0], bounding_boxes[i][1]), labels_in_image[i], fill="red")
        #     draw.rectangle(((bounding_boxes[i][0], bounding_boxes[i][1]), (bounding_boxes[i][2], bounding_boxes[i][3])), outline="red")
        # image.show()

        #save image with just the date in the name
        image_name = str(image_path).split("/")[-1]
        image.save("/caa/Homes01/mburges/viennaup23-hackathon-recycling/backend/example_images/{}".format(image_name))

        # print("boxes_nms", bounding_boxes.shape)
        # print("embeddings_nms", embeddings_nms.shape)
        # print("labels_nms", labels_in_image)

        patches = []
        for box in bounding_boxes:
            box = box.detach().cpu().numpy()
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            patches.append(image.crop((x0, y0, x0+w, y0+h)))

        # print cosine similarity between all pairs of bounding boxes torch
        for emb, label, class_id, patch in zip(embeddings_nms, labels_in_image, labels_nms, patches):
            feature_embeddings.append((emb, label, class_id, patch))

    return feature_embeddings