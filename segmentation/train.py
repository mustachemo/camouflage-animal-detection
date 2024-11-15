from transformers import AutoModelForImageSegmentation, AutoProcessor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import torch

# Load the pre-trained model
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "zhengpeng7/BiRefNet-COD", trust_remote_code=True
)

# Load the processor (for preprocessing images)
processor = AutoProcessor.from_pretrained("zhengpeng7/BiRefNet-COD")


# Define preprocessing
def preprocess_image(image, size=512):
    transforms = Compose([
        Resize((size, size)),
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    return transforms(image)


# Load your dataset (e.g., COD10k)
# Replace with your dataset loading mechanism
from datasets import load_dataset

dataset = load_dataset("path_to_cod10k")


# Preprocess dataset
def preprocess_data(example):
    image = example["image"]  # Adjust key to your dataset
    mask = example["mask"]  # Adjust key to your dataset
    image = preprocess_image(image)
    mask = preprocess_image(mask)  # Ensure mask has the same transformations
    return {"image": image, "mask": mask}


# Apply preprocessing
train_dataset = dataset["train"].map(preprocess_data)
val_dataset = dataset["validation"].map(preprocess_data)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)


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
        outputs = birefnet(pixel_values=images)
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


birefnet.save_pretrained("birefnet-cod10k-finetuned")
processor.save_pretrained("birefnet-cod10k-finetuned")


# Load test image
from PIL import Image

test_image = Image.open("path_to_test_image").convert("RGB")
test_image_preprocessed = preprocess_image(test_image).unsqueeze(0).to(device)

# Predict
birefnet.eval()
with torch.no_grad():
    outputs = birefnet(pixel_values=test_image_preprocessed)

# Post-process and visualize results
pred_mask = outputs.logits.argmax(dim=1).squeeze().cpu().numpy()
