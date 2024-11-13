import cv2
import os
import glob

def mask_dataset(original_image_path, mask_path, output_folder):
    """
    Clips the object from the original image using the mask and saves it in a labeled folder structure.

    :param original_image_path: Path to the original image
    :param mask_path: Path to the mask image
    :param output_folder: Folder to save the clipped image with a labeled structure
    :return: None
    """

    # Load the original image and the mask
    original_image = cv2.imread(original_image_path)
    mask = cv2.imread(mask_path, 0)  # Load mask in grayscale
    
    # Check if images were loaded successfully
    if original_image is None:
        print(f"Failed to load original image: {original_image_path}")
        return
    if mask is None:
        print(f"Failed to load mask image: {mask_path}")
        return
    
    print("Successfully loaded both original image and mask.")

    # Apply mask to the original image to isolate the object
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)

    # Create RGBA image
    b, g, r = cv2.split(masked_image)
    alpha_channel = mask
    rgba_image = cv2.merge((b, g, r, alpha_channel))
    print("RGBA image created.")

    # Extract class label from the original image filename
    filename = os.path.basename(original_image_path)
    parts = filename.split('-')
    
    # Use parts of the filename to determine the label
    if len(parts) >= 6:
        class_label = f"{parts[3]}-{parts[5]}"
    else:
        class_label = filename.split('.')[0]  # Fallback if unexpected format

    # Create the class-specific folder in the output directory
    class_folder = os.path.join(output_folder, class_label)
    if "COD10K" in class_folder:
        print(f"Skipping file: {filename} (contains 'COD10K')")
        return
    os.makedirs(class_folder, exist_ok=True)  # Create the folder if it doesn't exist

    
    
    # Define output path for the image within the class folder
    output_path = os.path.join(class_folder, f"{filename}")
    print(f"Saving to: {output_path}")

    

    # Save the result and check if saving is successful
    success = cv2.imwrite(output_path, rgba_image)
    if success:
        print(f"Successfully saved clipped image as {output_path}")
    else:
        print(f"Failed to save image: {output_path}")

# Set up paths
original_images_folder = "data/COD10K-v3/Train/Image"  # Folder containing .jpg images
mask_images_folder = "data/COD10K-v3/Train/GT_Object"  # Folder containing corresponding .png masks
output_folder = "data/Labeled_MaskDataset"                 # Output folder for labeled dataset

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each .jpg image in the original images folder
for original_image_path in glob.glob(os.path.join(original_images_folder, "*.jpg")):
    # Get the filename and construct the corresponding mask filename
    image_name = os.path.basename(original_image_path)
    mask_name = os.path.splitext(image_name)[0] + ".png"  # Replace .jpg with .png for mask

    # Construct the full path for the mask
    mask_path = os.path.join(mask_images_folder, mask_name)
    
    # Check if the mask file exists
    if os.path.exists(mask_path):
        mask_dataset(original_image_path, mask_path, output_folder)
    else:
        print(f"Mask not found for image: {image_name}")
