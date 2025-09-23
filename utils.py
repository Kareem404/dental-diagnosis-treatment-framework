import torch
from model import CompleteModel
from PIL import Image
import numpy as np
import io
import cv2
import math
import random
from huggingface_hub import hf_hub_download
import os

hf_token = os.getenv('access_token')

conditions = [
    "Cavitated",
    "Retained Root",
    "Crowned",
    "Filled",
    "Impacted",
    "Implant",
]
treatments = ["Filling", "Root Canal", "Extraction", "None"]

def byte_to_pixels(image_bytes):
    imageStream = io.BytesIO(image_bytes)
    return np.array(Image.open(imageStream))
    
def get_model():
    cropped_image_model_file = hf_hub_download(repo_id="Kareem-404/dental-diagnosis-and-treatment", 
                                                filename="epoch=253-step=9906.ckpt", 
                                                token=hf_token)

    sub_image_model_file = hf_hub_download(repo_id="Kareem-404/dental-diagnosis-and-treatment", 
                                            filename="epoch=342-step=13377.ckpt", 
                                            token=hf_token)
    
    ckpt_best = hf_hub_download(repo_id="Kareem-404/dental-diagnosis-and-treatment", 
                                filename="best_ckpt.pth", 
                                token=hf_token)
    
    ckpt = torch.load(ckpt_best)
    model = CompleteModel(cropped_image_model_file=cropped_image_model_file, sub_image_model_file=sub_image_model_file)
    model_state_dict = ckpt["model_state_dict"]
    model.load_state_dict(model_state_dict)
    return model

def process_prediction(logits):
    condition_logits, treatment_logits = logits
    condition_agrmax = torch.argmax(condition_logits, dim=1)
    treatment_argmax = torch.argmax(treatment_logits, dim=1)
    condition_preds = list(map(lambda x: conditions[x], condition_agrmax))
    treatment_preds = list(map(lambda x: treatments[x], treatment_argmax))
    return condition_preds, treatment_preds


def draw_rotated_box_opencv(image, x_percent, y_percent, width_percent, height_percent, rotation_degrees, color=(0, 255, 0), thickness=7, label=None):
    """
    Draw a rotated bounding box on an image using OpenCV.
    
    Args:
        image: OpenCV image (numpy array)
        x_percent: x-coordinate of top-left corner (percentage, 0-100)
        y_percent: y-coordinate of top-left corner (percentage, 0-100)
        width_percent: width of the box (percentage, 0-100)
        height_percent: height of the box (percentage, 0-100)
        rotation_degrees: rotation angle in degrees
        color: BGR color tuple for the box
        thickness: thickness of the box lines
        label: optional text label to draw
    
    Returns:
        image: annotated image
    """
    img_height, img_width = image.shape[:2]
    
    # Convert percentages to pixels
    x = (x_percent / 100) * img_width
    y = (y_percent / 100) * img_height
    width = (width_percent / 100) * img_width
    height = (height_percent / 100) * img_height
    
    # Calculate the four corners of the rotated rectangle
    # Using the same logic as your get_rotated_corner_points function
    rotation_radians = math.radians(rotation_degrees)
    cos_a = math.cos(rotation_radians)
    sin_a = math.sin(rotation_radians)
    
    # Corner points relative to top-left corner
    x1, y1 = x, y  # top left
    x2, y2 = x + width * cos_a, y + width * sin_a  # top right
    x3, y3 = x2 - height * sin_a, y2 + height * cos_a  # bottom right
    x4, y4 = x - height * sin_a, y + height * cos_a  # bottom left
    
    # Convert to integer coordinates
    points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
    
    # Draw the rotated rectangle
    cv2.polylines(image, [points], True, color, thickness)
    
    # Add label if provided
    if label:
        # Put label at top-left corner
        label_pos = (int(x1), int(y1) - 10)
        cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
    
    return image


def random_color():
    """Return a bright random BGR color for visibility on X-rays."""
    return (
        random.randint(50, 255),  # B
        random.randint(50, 255),  # G
        random.randint(50, 255),  # R
    )

def draw_bboxes(
        image, x_coords, y_coords, widths, heights, rotations, conditions, treatments
    ):
    # draw boxes
    annotated = image.copy()
    for i, (x, y, w, h, r) in enumerate(zip(x_coords, y_coords, widths, heights, rotations,)):
        color = random_color()
        annotated = draw_rotated_box_opencv(annotated, x, y, w, h, r, color=color, label=f"{i+1}")

    # add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    line_height = 45
    margin = 10

    # Prepare all text lines
    lines = [
        f"{i+1}. Condition: {cond}, Treatment: {treat}"
        for i, (cond, treat) in enumerate(zip(conditions, treatments))
    ]

    # Calculate background box size
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    max_width = max([tw for tw, th in text_sizes])
    total_height = line_height * len(lines)

    # Draw semi-transparent rectangle
    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (margin, margin),
        (margin + max_width + 10, margin + total_height + 10),
        (0, 0, 0),
        -1,
    )
    alpha = 0.5  # transparency
    annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)

    # Put each line of text
    for i, line in enumerate(lines):
        y_offset = margin + 25 + i * line_height
        cv2.putText(
            annotated,
            line,
            (margin + 5, y_offset),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated