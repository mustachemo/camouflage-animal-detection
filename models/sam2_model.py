import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import cv2
from pathlib import Path

INPUT_DIR = Path("static/sample_data/input")
OUTPUT_DIR = Path("static/sample_data/output")


def get_input_images():
    for image in INPUT_DIR.glob("*.jpg"):
        yield cv2.imread(str(image)), image.stem


checkpoint = "sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    # image = cv2.imread("data/test/COD10K-CAM-1-Aquatic-1-BatFish-8.jpg")

    for image, name in get_input_images():
        center_x, center_y = image.shape[0] // 2, image.shape[1] // 2
        predictor.set_image(image)

        input_point = np.array([[center_x, center_y]])
        input_label = np.array([0])
        masks, _, _ = predictor.predict(
            # point_coords=input_point,
            point_labels=input_label,
            box=np.array([0, 800, 418, 158]),
            multimask_output=True,
        )

        for idx, mask in enumerate(masks):
            print(
                f"mask.shape: {mask.shape} | mask.dtype: {mask.dtype} | mask.max(): {mask.max()} | mask.min(): {mask.min()}"
            )
            cv2.imwrite(
                str(OUTPUT_DIR / f"{name}_mask{idx}.jpg"), (mask * 255).astype(np.uint8)
            )
