import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from pathlib import Path
import torch

model = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,  # model output channels (number of classes in your dataset)
)
preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')


TEST_IMAGES = Path("data/test/images")
TEST_MASKS = Path("data/test/masks")
OUTPUT_DIR = Path("data/test/pretrained_outputs")


def dice_score(pred_mask, true_mask, epilson=1e-6):
    intersection = torch.logical_and(pred_mask, true_mask).sum().item()
    union = pred_mask.sum().item() + true_mask.sum().item()
    return 2 * intersection / union + epilson


if __name__ == "__main__":
    