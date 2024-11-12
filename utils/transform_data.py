"""
Put "COD10K-v3" dataset in the data folder and run this script to transform the data into
the format required by the model.
"""

from pathlib import Path
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

INPUT_DIR = Path("/home/mustachemo/Data/COD10K-v3")
OUTPUT_DIR = Path.cwd() / "./data/"

resize_transform = transforms.Resize((512, 512))


def create_and_clean_dir(dir_path):
    if dir_path.exists():
        shutil.rmtree(str(dir_path))
    (dir_path / "train" / "images").mkdir(parents=True, exist_ok=True)
    (dir_path / "train" / "masks").mkdir(parents=True, exist_ok=True)
    (dir_path / "test" / "images").mkdir(parents=True, exist_ok=True)
    (dir_path / "test" / "masks").mkdir(parents=True, exist_ok=True)


def copy_and_resize_files(file_paths, src_dir, dst_dir, is_mask=False):
    for file_path in tqdm(
        file_paths, desc=f"Copying files to {dst_dir.split('/')[-1]}", unit="files"
    ):
        img = Image.open(src_dir / file_path)
        img = resize_transform(img)

        if is_mask:
            img = img.convert("L")  # Convert to grayscale
            img = np.array(img)
            img = (img > 0).astype(np.uint8) * 255  # Binarize the mask
            img = Image.fromarray(img)

        img.save(dst_dir / file_path.name)


def main(input_dir, output_dir):
    create_and_clean_dir(output_dir)

    image_paths = list((input_dir / "Train" / "Image").glob("*")) + list(
        (input_dir / "Test" / "Image").glob("*")
    )
    mask_paths = list((input_dir / "Train" / "GT_Object").glob("*")) + list(
        (input_dir / "Test" / "GT_Object").glob("*")
    )

    # Create a list of tuples (image_path, mask_path)
    paired_paths = [
        (img, mask) for img, mask in zip(sorted(image_paths), sorted(mask_paths))
    ]

    print(f"length of paired_paths: {len(paired_paths)}")

    train_pairs, test_pairs = train_test_split(
        paired_paths, test_size=0.1, random_state=42
    )

    # Unzip the pairs
    image_train, mask_train = zip(*train_pairs)
    image_test, mask_test = zip(*test_pairs)

    copy_and_resize_files(
        image_train, input_dir / "Train" / "Image", output_dir / "train" / "images"
    )
    copy_and_resize_files(
        image_test, input_dir / "Test" / "Image", output_dir / "test" / "images"
    )
    copy_and_resize_files(
        mask_train,
        input_dir / "Train" / "GT_Object",
        output_dir / "train" / "masks",
        is_mask=True,
    )
    copy_and_resize_files(
        mask_test,
        input_dir / "Test" / "GT_Object",
        output_dir / "test" / "masks",
        is_mask=True,
    )


if __name__ == "__main__":
    main(INPUT_DIR, OUTPUT_DIR)
