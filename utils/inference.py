import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import pandas as pd

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations (should match the ones used in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load labels from CSV
labels_df = pd.read_csv("data/Species_Labels.csv")  # Assuming labels.csv has one column 'label'
labels = labels_df['Label'].tolist()

# Load the trained model and modify the final layer for 74 classes
model = models.resnet50(weights=None)  # Initialize model without pretrained weights
model.fc = nn.Linear(model.fc.in_features, len(labels))  # Adjust for number of classes
model.load_state_dict(torch.load("fine_tuned_model.pth", map_location=device, weights_only=True))
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Function to make predictions and get label index
def predict(image_path):
    # Load image and apply transformations
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move image to the device and make prediction
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted_index = output.max(1)  # Get the index of the highest score
    
    return predicted_index.item(), labels[predicted_index.item()]

# Example usage
image_path = "clipped_object.png"  # Replace with your image path
predicted_index, predicted_label = predict(image_path)
print(f"Predicted index: {predicted_index}")
print(f"Predicted label: {predicted_label}")
