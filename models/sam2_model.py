import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np

checkpoint = "sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image("data/test/COD10K-CAM-1-Aquatic-1-BatFish-8.pjg")
    
    
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    masks, _, _ = predictor.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
