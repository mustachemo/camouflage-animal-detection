from torchvision.io import read_image
from torchvision.models.segmentation import (
    fcn_resnet50,
    FCN_ResNet50_Weights,
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale
from pathlib import Path
import torch
from tqdm import tqdm
from PIL import Image

TEST_IMAGES = Path("data/test/images")
TEST_MASKS = Path("data/test/masks")
OUTPUT_DIR = Path("data/test/pretrained_outputs")


def dice_score(pred_mask, true_mask, epilson=1e-6):
    intersection = torch.logical_and(pred_mask, true_mask).sum().item()
    union = pred_mask.sum().item() + true_mask.sum().item()
    return 2 * intersection / union + epilson


transform = Compose([Resize((520, 520)), ToTensor()])

if __name__ == "__main__":
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    dice_score_value = []
    for image_path in tqdm(
        TEST_IMAGES.glob("*.jpg"), total=len(list(TEST_IMAGES.glob("*.jpg")))
    ):
        image = Image.open(str(image_path))
        # save the original dimensions
        original_width, original_height = image.size

        filename = image_path.stem
        mask_path = TEST_MASKS / f"{filename}.png"
        mask = Image.open(str(mask_path))

        # resize both image and mask to 256x256 using transform
        image = transform(image)
        mask = transform(mask).squeeze(0)

        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)["out"]
            print(f"1. output.shape {output.shape}")
            output_predictions = output.argmax(1).squeeze(0)
            print(f"2. output.shape {output_predictions.shape}")
            # output_predictions = torch.sigmoid(output).squeeze(0)

        # Apply threshold to binarize the predictions
        threshold = 0.5
        output_predictions = (output_predictions > threshold).float()

        print(
            f"output_precdiction max min {output_predictions.max()} {output_predictions.min()}"
        )
        print(f"output.shape {output_predictions.shape}")

        dice = dice_score(output_predictions, mask)
        print(f"Dice Score: {dice}")
        dice_score_value.append(dice)

        # Resize the output to the original dimensions
        output_predictions = to_pil_image(output_predictions)
        output_predictions = output_predictions.resize((
            original_width,
            original_height,
        ))
        output_predictions.save(OUTPUT_DIR / f"{filename}.png")

    print(f"Average Dice Score: {sum(dice_score) / len(dice_score)}")
