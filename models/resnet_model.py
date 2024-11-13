# Imports
from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from PIL import Image
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from torchmetrics.classification import MulticlassAccuracy

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
    print(f"Dataset loaded with {len(dataset)} samples and {len(dataset.classes)} classes.")

    # Split dataset into training and validation sets
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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

    # Initialize metric calculators
    accuracy_calculator = MulticlassAccuracy(num_classes=num_classes).to(device)

    # Dictionary to store metrics
    metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }

    # Training and validation loop
    num_epochs = 20
    print("Starting training and validation loop...")

    for epoch in range(num_epochs):
        # Training phase
        running_loss = 0.0
        running_accuracy = 0.0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Zero gradients, forward pass, compute loss, backprop, optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            running_accuracy += accuracy_calculator(outputs, labels).item()

        # Calculate and log training metrics for this epoch
        avg_train_loss = running_loss / len(train_loader)
        avg_train_accuracy = running_accuracy / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Track validation loss and accuracy
                val_loss += loss.item()
                val_accuracy += accuracy_calculator(outputs, labels).item()

                # Store labels and predictions for precision, recall, and F1 calculation
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Calculate and log validation metrics for this epoch
        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_accuracy / len(val_loader)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        # Save metrics
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(avg_train_loss)
        metrics['val_loss'].append(avg_val_loss)
        metrics['train_accuracy'].append(avg_train_accuracy)
        metrics['val_accuracy'].append(avg_val_accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Train Accuracy: {avg_train_accuracy:.4f}, Val Accuracy: {avg_val_accuracy:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    print("Training and validation complete.")
    torch.save(model.state_dict(), "fine_tuned_model.pth")
    print("Model saved as 'fine_tuned_model.pth'.")

    # Save metrics to file
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("training_validation_metrics.csv", index=False)
    print("Metrics saved as 'training_validation_metrics.csv'.")


