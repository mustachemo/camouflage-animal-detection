import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import cv2

checkpoint = "sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    image = cv2.imread("data/test/COD10K-CAM-1-Aquatic-1-BatFish-8.jpg")
    center_x, center_y = image.shape[0] // 2, image.shape[1] // 2
    # image = cv2.imread("data/test/truck.jpg")
    predictor.set_image(image)
    
    
    # input_point = np.array([[500, 375]])
    input_point = np.array([[center_x, center_y]])
    input_label = np.array([0])
    masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )


    for mask in masks:
        cv2.imshow("mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()