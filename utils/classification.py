import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import pandas as pd
import cv2
import numpy as np

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations (should match the ones used in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load labels from CSV
labels_df = pd.read_csv("./configs/Species_Labels_1.csv")
labels = labels_df["Label"].tolist()

# Load the trained model and modify the final layer for 74 classes
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(labels))
model.load_state_dict(
    torch.load("fine_tuned_model.pth", map_location=device, weights_only=True)
)
model = model.to(device)
model.eval()


# Function to clip object from original image using the mask
def clip_object(original_image, mask):
    # Resize mask to match the size of the original image if necessary
    if mask.shape[:2] != original_image.shape[:2]:
        mask = cv2.resize(
            mask,
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Ensure mask is a single channel grayscale image
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Apply the mask
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)

    # Create an alpha channel using the mask
    b, g, r = cv2.split(masked_image)
    alpha_channel = mask
    rgba_image = cv2.merge((b, g, r, alpha_channel))

    return rgba_image


# Function to make predictions on the clipped object
def predict_clipped_object(original_image_path, mask_path=None):
    # Load images
    original_image = cv2.imread(original_image_path)
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Clip object and save the result temporarily
        clipped_image = clip_object(original_image, mask)
        clipped_image_path = "clipped_object.png"
        cv2.imwrite(clipped_image_path, clipped_image)

    else:
        clipped_image_path = original_image_path

    image = Image.open(clipped_image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    # Make prediction
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted_index = output.max(1)

    return predicted_index.item(), labels[predicted_index.item()]


# Example usage
# original_image_path = "frames_output/frame_0010.jpg"
# mask_path = "frames_output/frame_0010.jpg"

# # Without mask
# predicted_index, predicted_label = predict_clipped_object(original_image_path)

# print(f"1.) Predicted index: {predicted_index}")
# print(f"Predicted label: {predicted_label}")
# # With mask
# predicted_index, predicted_label = predict_clipped_object(original_image_path, mask_path)

# print(f"2.)Predicted index: {predicted_index}")
# print(f"Predicted label: {predicted_label}")
