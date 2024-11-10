"""
Put "COD10K-v3" dataset in the data folder and run this script to transform the data into
the format required by the model.
"""

from pathlib import Path
import shutil
from tqdm import tqdm

INPUT_TRAINING_DIR = Path.cwd() / "./data/COD10K-v3/Train"
INPUT_TEST_DIR = Path.cwd() / "./data/COD10K-v3/Test"
OUTPUT_DIR = Path.cwd() / "./data/training"


def create_and_clean_dir(dir_path):
    if dir_path.exists():
        shutil.rmtree(str(dir_path))
    (dir_path / "images").mkdir(parents=True, exist_ok=True)
    (dir_path / "masks").mkdir(parents=True, exist_ok=True)


def move_files(training_dir, test_dir, output_dir):
    training_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    def move_directory_files(src_dir, dst_dir):
        src_path = Path(src_dir)
        dst_path = Path(dst_dir)
        for file_path in tqdm(
            src_path.glob("*"),
            desc=f"Moving files from {src_path.parent} to {dst_path.parent}",
            unit="files",
        ):
            if file_path.is_file():
                shutil.copy(str(file_path), str(dst_path / file_path.name))

    # move images
    move_directory_files(training_dir / "Image", output_dir / "images")
    move_directory_files(test_dir / "Image", output_dir / "images")
    # move masks
    move_directory_files(training_dir / "GT_Object", output_dir / "masks")
    move_directory_files(test_dir / "GT_Object", output_dir / "masks")


if __name__ == "__main__":
    create_and_clean_dir(OUTPUT_DIR)
    move_files(INPUT_TRAINING_DIR, INPUT_TEST_DIR, OUTPUT_DIR)
