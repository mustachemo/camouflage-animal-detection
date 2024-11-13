#Used to create a fine tuned model for the Mask Dataset
from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import os

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations with normalization
print("Defining transformations...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing to match input size of common pretrained models
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ImageNet models
])

# Load dataset
print("Loading dataset...")
data_dir = "data/Labeled_MaskDataset"  # Ensure this path is correct and accessible
if not os.path.isdir(data_dir):
    print(f"Dataset directory '{data_dir}' not found.")
else:
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Dataset loaded with {len(dataset)} samples and {len(dataset.classes)} classes.")

    # Load pretrained model
    print("Loading pretrained model...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = model.to(device)  # Move model to device

    # Modify the final layer for your number of classes
    num_classes = len(dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    print(f"Model modified for {num_classes} classes.")

    # Unfreeze more layers for better fine-tuning
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():  # Unfreeze more layers to improve learning
        param.requires_grad = True
    for param in model.layer4.parameters():  # Unfreeze even the last layers
        param.requires_grad = True

    # Loss and optimizer with a modified learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)  # Reduce learning rate

    # Training loop with debugging output
    num_epochs = 20
    print("Starting training loop...")
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # Move data to device

            # Zero gradients, forward pass, compute loss, backprop, optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate loss for display
            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

    print("Training complete.")
    torch.save(model.state_dict(), "fine_tuned_model.pth")
    print("Model saved as 'fine_tuned_model.pth'.")