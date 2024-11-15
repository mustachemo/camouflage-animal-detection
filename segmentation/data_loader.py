import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor


class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(512, 512)):
        """
        Args:
            image_dir (str): Directory containing images.
            mask_dir (str): Directory containing masks.
            image_size (tuple): Desired size of images and masks (width, height).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.image_transform = Compose([
            Resize(image_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = Compose([
            Resize(image_size),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Assuming masks are grayscale

        # Apply transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        return {"image": image, "mask": mask}
