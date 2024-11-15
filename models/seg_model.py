from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

birefnet = None
device = None
transform_image = None


def initialize_seg_model():
    global birefnet, device, transform_image

    birefnet = AutoModelForImageSegmentation.from_pretrained(
        "zhengpeng7/BiRefNet-COD", trust_remote_code=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    birefnet.to(device)
    birefnet.eval()

    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_mask(image_path):
    global birefnet, device, transform_image

    image = Image.open(image_path).convert("RGB")
    input_images = transform_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()

    pred_pil = transforms.ToPILImage()(pred)
    return pred_pil


# Example usage:
# initialize_model()
# mask = get_mask("path_to_your_image.jpg")
