from torchvision.io import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

TRAIN_IMAGES = Path("data/train/images")
TRAIN_MASKS = Path("data/train/masks")


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


def dice_score(pred_mask, true_mask, epsilon=1e-6):
    intersection = (pred_mask * true_mask).sum().item()
    union = pred_mask.sum().item() + true_mask.sum().item()
    return (2 * intersection) / (union + epsilon)


weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights)
model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1))

optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.BCEWithLogitsLoss()  # For binary classification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_dice_score = 0
    for image_path in tqdm(
        TRAIN_IMAGES.glob("*.jpg"), desc=f"Epoch {epoch + 1}/{num_epochs}"
    ):
        # Load and preprocess image and mask
        image = read_image(str(image_path)).float().div(255).to(device)
        mask_path = TRAIN_MASKS / f"{image_path.stem}.png"
        mask = read_image(str(mask_path)).float().div(255).to(device)

        # Add batch dimension
        input_tensor = image.unsqueeze(0)  # Shape [1, 3, H, W]
        mask = mask.unsqueeze(0)  # Shape [1, 1, H, W]

        # Forward pass
        optimizer.zero_grad()
        output = model(input_tensor)["out"]  # Shape [1, 1, H, W]

        print(f"output.shape {output.shape}")
        print(f"output.max() {output.max()}")
        print(f"output.min() {output.min()}")
        print(f"mask.shape {mask.shape}")
        print(f"mask.max() {mask.max()}")
        print(f"mask.min() {mask.min()}")

        # Calculate loss
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()

        # Calculate Dice score
        with torch.no_grad():
            output_predictions = torch.sigmoid(output)  # Apply sigmoid
            output_predictions = (
                output_predictions > 0.5
            ).float()  # Threshold at 0.5 for binary mask
            dice = dice_score(output_predictions, mask)
            total_dice_score += dice

    # Average Dice score per epoch
    avg_dice_score = total_dice_score / len(list(TRAIN_IMAGES.glob("*.jpg")))
    print(
        f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Avg Dice Score: {avg_dice_score:.4f}"
    )

    # save each epoch model weights
    torch.save(
        model.state_dict(), f"checkpoints/fcn_resent50_model_epoch_{epoch + 1}.pth"
    )

print("Training complete.")
