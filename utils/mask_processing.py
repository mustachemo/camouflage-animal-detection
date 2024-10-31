import cv2
import numpy as np


def clip_object(original_image, mask):
    """
    Clips the object from the original image using the mask.

    :param original_image: The original image
    :param mask: The mask of the object
    :return: The clipped object
    """
    # Create a mask with 3 channels to match the original image
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)

    # Add an alpha channel to the original image
    b, g, r = cv2.split(masked_image)
    alpha_channel = mask
    rgba_image = cv2.merge((b, g, r, alpha_channel))

    return rgba_image
