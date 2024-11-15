from transformers import AutoModelForImageSegmentation, AutoProcessor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import torch
from data_loader import ImageMaskDataset

# Load the pre-trained model
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "zhengpeng7/BiRefNet-COD", trust_remote_code=True
)

# Load your dataset (e.g., COD10k)
# Replace with your dataset loading mechanism
train_dataset = ImageMaskDataset(
    image_dir="./data/train/images",
    mask_dir="./data/train/masks",
    image_size=(512, 512),  # Resize all images and masks to 512x512
)


# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)


from torch.optim import AdamW
from torch.nn import functional as F

# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = AdamW(birefnet.parameters(), lr=5e-5)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
birefnet.to(device)


epochs = 5

for epoch in range(epochs):
    birefnet.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        # Move data to device
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # Forward pass
        outputs = birefnet(images)
        loss = criterion(outputs.logits, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

    # Validation step
    birefnet.eval()
    with torch.no_grad():
        val_loss = 0
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = birefnet(pixel_values=images)
            val_loss += criterion(outputs.logits, masks).item()

    print(f"Validation Loss: {val_loss / len(val_loader)}")


try:
    birefnet.save_pretrained("birefnet-cod10k-finetuned")
except FileNotFoundError:
    print("Model not saved. Please check the path.")
