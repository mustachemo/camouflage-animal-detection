import os
import sys
from PIL import Image
import csv
from datetime import datetime

def process_images(directory, target_width, target_height):
    # Create CSV file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"image_resize_log_{timestamp}.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['Directory', 'Original Filename', 'Original Dimensions', 'New Filename'])
        
        # Walk through all directories
        for root, dirs, files in os.walk(directory):
            # Filter for jpg files (case insensitive)
            jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg'))]
            
            for filename in jpg_files:
                try:
                    # Full path to original file
                    original_path = os.path.join(root, filename)
                    
                    # Open and get original image dimensions
                    with Image.open(original_path) as img:
                        original_width, original_height = img.size
                        original_dimensions = f"{original_width}x{original_height}"
                        
                        # Resize image
                        resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                        
                        # Create new filename
                        name_without_ext = os.path.splitext(filename)[0]
                        extension = os.path.splitext(filename)[1]
                        new_filename = f"{name_without_ext}_{target_width}x{target_height}{extension}"
                        new_path = os.path.join(root, new_filename)
                        
                        # Save resized image
                        resized_img.save(new_path, quality=95)
                        
                        # Log the change
                        csv_writer.writerow([root, filename, original_dimensions, new_filename])
                        
                        print(f"Processed: {filename} -> {new_filename}")
                        
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py width height")
        print("Example: python script.py 800 600")
        sys.exit(1)
        
    try:
        target_width = int(sys.argv[1])
        target_height = int(sys.argv[2])
    except ValueError:
        print("Width and height must be integers")
        sys.exit(1)
        
    # Start from current directory
    current_dir = os.getcwd()
    process_images(current_dir, target_width, target_height)
    
if __name__ == "__main__":
    main()
