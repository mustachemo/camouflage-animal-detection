from torchvision.io import read_image
from torchvision.models.segmentation import (
    fcn_resnet101,
    FCN_ResNet101_Weights,
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights,
)
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

TRAIN_IMAGES = Path("data/train/images")
TRAIN_MASKS = Path("data/train/masks")
CHECKPOINTS_DIR = Path("checkpoints")
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def save_training_log_and_model_params(training_log, model, csv_path):
    log_df = pd.DataFrame(training_log)

    model_params = {
        f"param_{name}": param.item()
        for name, param in model.state_dict().items()
        if param.numel() == 1
    }

    model_params_df = pd.DataFrame([model_params])
    final_df = pd.concat([log_df, model_params_df], axis=1)
    final_df.to_csv(csv_path, index=False)
    print(f"Training log and model parameters saved to {csv_path}")


# Dice score and Dice loss
def dice_score(pred_mask, true_mask, epsilon=1e-6):
    intersection = (pred_mask * true_mask).sum().item()
    union = pred_mask.sum().item() + true_mask.sum().item()
    return (2 * intersection) / (union + epsilon)


def dice_loss(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice_loss = 1 - (2 * intersection + epsilon) / (union + epsilon)
    return dice_loss


weights = FCN_ResNet101_Weights.DEFAULT
model = fcn_resnet101(weights=weights)
model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1))

# optimizer = optim.Adam(model.parameters(), lr=3e-4)
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()  # For binary classification
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "max", factor=0.33, patience=3, min_lr=1e-7
)  # goal: maximize Dice score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

n_total = len(list(TRAIN_IMAGES.glob("*.jpg")))

# Training loop
num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    total_dice_score = 0
    epoch_loss = 0
    with tqdm(
        total=n_total, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="img"
    ) as pbar:
        for image_path in TRAIN_IMAGES.glob("*.jpg"):
            # Load and preprocess image and mask
            image = read_image(str(image_path)).float().div(255).to(device)
            mask_path = TRAIN_MASKS / f"{image_path.stem}.png"
            mask = read_image(str(mask_path)).float().div(255).to(device)

            # Add batch dimension
            input_tensor = image.unsqueeze(0)  # Shape [1, 3, H, W]
            mask = mask.unsqueeze(0)  # Shape [1, 1, H, W]

            # Forward pass
            optimizer.zero_grad(set_to_none=True)
            output = model(input_tensor)["out"]  # Shape [1, 1, H, W]

            # Calculate loss
            bce_loss = criterion(output, mask)
            dice = dice_loss(torch.sigmoid(output), mask)
            loss = bce_loss + dice

            loss.backward()
            optimizer.step()

            pbar.update(1)
            epoch_loss += loss.item()
            pbar.set_postfix(**{"loss (batch)": loss.item()})

            # Calculate Dice score
            with torch.no_grad():
                output_predictions = torch.sigmoid(output)  # Apply sigmoid
                output_predictions = (
                    output_predictions > 0.5
                ).float()  # Threshold at 0.5 for binary mask
                dice = dice_score(output_predictions, mask)
                total_dice_score += dice

    # Average Dice score per epoch
    avg_dice_score = total_dice_score / n_total
    print(
        f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Avg Dice Score: {avg_dice_score:.4f}"
    )
    # Adjust learning rate based on Dice score
    scheduler.step(avg_dice_score)
    checkpoint_path = CHECKPOINTS_DIR / f"fcn_resnet101_model_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

print("Training complete.")
