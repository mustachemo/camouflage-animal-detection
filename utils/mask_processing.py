import cv2
import numpy as np
import os 
import glob

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

##Test Code:
# original_image = cv2.imread("data/CAMO-V.1.0-CVIU2019/Images/Test/camourflage_01244.jpg")
# mask = cv2.imread("data/CAMO-V.1.0-CVIU2019/GT/camourflage_01244.png", cv2.IMREAD_GRAYSCALE)

# mask = clip_object(original_image, mask)

# cv2.imwrite("clipped_object.png", mask)

