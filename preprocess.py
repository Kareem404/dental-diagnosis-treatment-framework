import torch
import numpy as np
import cv2
from torchvision import transforms
import math


def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            min_max_normalize, # function to apply min_max normalization to image
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
)


def get_binary_mask(image, x, y, width, height, rotation):
    """Returns binary mask of the image with the rotated bounding box as the region of interest."""
    img_width = image.shape[1]
    img_height = image.shape[0]

    w, h = width * img_width / 100, height * img_height / 100
    a = math.pi * (rotation / 180.0) if rotation else 0.0
    cos_a, sin_a = math.cos(a), math.sin(a)

    x1, y1 = x * img_width / 100, y * img_height / 100  # top left
    x2, y2 = x1 + w * cos_a, y1 + w * sin_a  # top right
    x3, y3 = x2 - h * sin_a, y2 + h * cos_a  # bottom right
    x4, y4 = x1 - h * sin_a, y1 + h * cos_a  # bottom left

    coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.fillPoly(mask, np.array([coords], dtype=np.int32), 255)
    return mask

def get_sub_image(image, x, y, width, height, rotation):
    # rotation *= -1
    rotation_copy = rotation * -1
    original_height, original_width = image.shape[:2]
    pixel_x = x / 100.0 * original_width
    pixel_y = y / 100.0 * original_height
    pixel_width = width / 100.0 * original_width
    pixel_height = height / 100.0 * original_height
    center_x = pixel_x + pixel_width / 2
    center_y = pixel_y + pixel_height / 2

    rotation_matrix = np.array(
        [
            [np.cos(np.radians(rotation_copy)), -np.sin(np.radians(rotation_copy)), 0],
            [np.sin(np.radians(rotation_copy)), np.cos(np.radians(rotation_copy)), 0],
            [0, 0, 1],
        ]
    )

    translation_matrix_to_origin = np.array(
        [[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]]
    )
    translation_matrix_back = np.array(
        [[1, 0, pixel_width / 2], [0, 1, pixel_height / 2], [0, 0, 1]]
    )

    affine_matrix = np.dot(rotation_matrix, translation_matrix_to_origin)
    affine_matrix = np.dot(affine_matrix, translation_matrix_back)

    rotated = cv2.warpPerspective(
        image, affine_matrix, (int(pixel_width), int(pixel_height))
    )
    return rotated

def get_normalized_teeth_coords(image, x, y, width, height, rotation):
    img_width = image.shape[1]
    img_height = image.shape[0]

    w, h = width * img_width / 100, height * img_height / 100
    a = math.pi * (rotation / 180.0) if rotation else 0.0
    cos_a, sin_a = math.cos(a), math.sin(a)

    x1, y1 = x * img_width / 100, y * img_height / 100  # top left
    x2, y2 = x1 + w * cos_a, y1 + w * sin_a  # top right
    x3, y3 = x2 - h * sin_a, y2 + h * cos_a  # bottom right
    x4, y4 = x1 - h * sin_a, y1 + h * cos_a  # bottom left

    return (
        x1 / img_width,
        y1 / img_height,
        x2 / img_width,
        y2 / img_height,
        x3 / img_width,
        y3 / img_height,
        x4 / img_width,
        y4 / img_height,
    )

def get_cropped_image(image, x, y, width, height, zoom_factor=5):
    """Inputs x,y,width,height are in percentage (0-100)"""

    # Convert percentage values to pixel values
    img_height, img_width = image.shape[:2]
    x = int(x / 100.0 * img_width)
    y = int(y / 100.0 * img_height)
    width = int(width / 100.0 * img_width)
    height = int(height / 100.0 * img_height)
    # Calculate the bounding box around the target
    new_x = max(x - width // 2, 0)
    new_y = max(y - height // 2, 0)
    new_width = int(min(width * zoom_factor, img_width - new_x))
    new_height = int(min(height * zoom_factor, img_height - new_y))
    # Crop the image
    cropped_image = image[new_y : new_y + new_height, new_x : new_x + new_width]
    return cropped_image

# preprocess a row
# we basically have to loop and process xywhr alone
def preprocess_box(grayscale_img, x, y, width, height, rotation):

    # make sure image is grayscale if not
    # grayscale_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE) # read the image as grey-scale using cv2

    sub_image = get_sub_image(grayscale_img, x, y, width, height, rotation)

    '''coords = torch.tensor(
        get_normalized_teeth_coords(
            grayscale_img, x, y, width, height, rotation
        ),
        dtype=torch.float32
    )'''

    binary_mask = get_binary_mask(grayscale_img, x, y, width, height, rotation)

    grayscale_img = get_cropped_image(
        grayscale_img, x, y, width, height, zoom_factor=3
    )  # crop the image around the tooth (or prosthetic) with some zoom factor -> 2nd channel

    binary_mask = get_cropped_image(
        binary_mask, x, y, width, height, zoom_factor=3
    )

    sub_image = torch.from_numpy(sub_image).unsqueeze(0).float()
    sub_image = transform(sub_image)

    grayscale_img = torch.from_numpy(grayscale_img).unsqueeze(0).float()
    grayscale_img = transform(grayscale_img)

    binary_mask = torch.from_numpy(binary_mask).unsqueeze(0).float()
    binary_mask = transform(binary_mask)
    img_data = torch.cat([sub_image, grayscale_img, binary_mask], dim=0).to(dtype=torch.float32)

    return img_data

def yolo_obb_to_original_format(image_width, image_height, x_center, y_center, width, height, rotation_radians):
    """
    Convert YOLO OBB format back to original format.
    
    Args:
        image_width (int): Width of the image in pixels
        image_height (int): Height of the image in pixels
        x_center (float): x-coordinate of box center in pixels
        y_center (float): y-coordinate of box center in pixels
        width (float): width of the box in pixels
        height (float): height of the box in pixels
        rotation_radians (float): Rotation angle in radians
    
    Returns:
        tuple: (x, y, width, height, rotation_degrees)
            - x: x-coordinate of top-left corner (percentage, 0-100)
            - y: y-coordinate of top-left corner (percentage, 0-100)
            - width: width of the box (percentage, 0-100)
            - height: height of the box (percentage, 0-100)
            - rotation_degrees: rotation angle in degrees
    """
    
    # rotation angle in degrees
    rotation_degrees = math.degrees(rotation_radians)

    offset_x = -width / 2
    offset_y = -height / 2

    # rotate the offset
    x_top_left = x_center + offset_x * math.cos(rotation_radians) - offset_y * math.sin(rotation_radians)
    y_top_left = y_center + offset_x * math.sin(rotation_radians) + offset_y * math.cos(rotation_radians)

    # convert to percentages
    x_percentage = (x_top_left / image_width) * 100
    y_percentage = (y_top_left / image_height) * 100
    width_percentage = (width / image_width) * 100
    height_percentage = (height / image_height) * 100

    return x_percentage, y_percentage, width_percentage, height_percentage, rotation_degrees

def preprocess_image(image, boxes):
    all_img_data = []
    all_boxes = []
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for box in boxes:
        box_original = yolo_obb_to_original_format(grayscale_img.shape[1], grayscale_img.shape[0], *(box.cpu()))
        all_boxes.append(box_original)
        img_data = preprocess_box(grayscale_img, *box_original )
        all_img_data.append(img_data)
    return np.array(all_img_data), np.array(all_boxes)