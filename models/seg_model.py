import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

def initialize_seg_model():

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

    return birefnet, device, transform_image

def get_mask(image, birefnet, device, transform_image):
    

    input_images = transform_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()

    pred_pil = transforms.ToPILImage()(pred)
    return pred_pil
