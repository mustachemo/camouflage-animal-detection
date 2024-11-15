import os
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from PIL import Image
import numpy as np
from sklearn.metrics import jaccard_score
from pathlib import Path
from tqdm import tqdm

def initialize_seg_model():
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        "zhengpeng7/BiRefNet-COD", trust_remote_code=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    birefnet.to(device)
    birefnet.eval()

    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return birefnet, device, transform_image

def get_mask(image, birefnet, device, transform_image):
    input_images = transform_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    return pred_pil

def dice_score(pred_mask, true_mask):
    pred_mask = np.array(pred_mask).astype(bool)
    true_mask = np.array(true_mask).astype(bool)
    intersection = np.logical_and(pred_mask, true_mask).sum()
    return 2 * intersection / (pred_mask.sum() + true_mask.sum())

def main():
    # Initialize the model
    birefnet, device, transform_image = initialize_seg_model()
    
    # Directories
    image_dir = Path("./data/test/images")
    mask_dir = Path("./data/test/masks")
    
    
    # Calculate Dice scores
    dice_scores = []
    
    for image_path in tqdm(image_dir.glob("*.jpg"), total=len(list(image_dir.glob("*.jpg")))):
        filename = image_path.stem
        image_path = image_dir / f"{filename}.jpg"
        mask_path = mask_dir / f"{filename}.png"

        # Load image and true mask
        image = Image.open(image_path).convert("RGB")
        true_mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale for binary
        
        # Generate predicted mask
        pred_mask = get_mask(image, birefnet, device, transform_image)
        
        # Calculate Dice score and store it
        dice = dice_score(pred_mask, true_mask)
        dice_scores.append(dice)
    
    # Calculate and print average Dice score
    avg_dice_score = np.mean(dice_scores)
    print(f"Average Dice Score: {avg_dice_score:.4f}")

if __name__ == "__main__":
    main()
