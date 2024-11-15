import cv2
import numpy as np
import base64
from models.seg_model import SAM2Model


def detect_camouflaged_animals(image_data):
    """
    Detects camouflaged animals in the input image.

    Args:
        image_data: Image file uploaded by user (base64-encoded)

    Returns:
        Tuple of original image and the image with detected camouflaged animals.
    """
    image = decode_image(
        image_data
    )  # might be useful to keep the original image for later use

    model = SAM2Model()

    mask = model.segment(image)

    overlay = apply_mask(image, mask)  # ! can overlay the mask on the original image

    # return image, overlay
    return NotImplementedError


def decode_image(image_data):
    # decode the base64 image data (from Dash upload)
    encoded_data = image_data.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # return img
    return NotImplementedError


def apply_mask(image, mask):
    """Overlay the segmentation mask on the image."""
    overlay = image.copy()
    overlay[mask == 1] = [0, 255, 0]  # Example: Use green for detected areas
    # return overlay
    return NotImplementedError
