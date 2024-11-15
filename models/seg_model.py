# Imports
from PIL import Image
import torch
from torchvision import transforms
import os
from glob import glob


# Load Model
# Option 2 and Option 3 is better for local running -- we can modify codes locally.

# # # Option 1: loading BiRefNet with weights:
from transformers import AutoModelForImageSegmentation

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "zhengpeng7/BiRefNet-COD", trust_remote_code=True
)

# Option-2: loading weights with BiReNet codes:
# birefnet = BiRefNet.from_pretrained(
#     [
#         "zhengpeng7/BiRefNet-COD",
#     ][0]
# )

# # Option-3: Loading model and weights from local disk:
# from utils import check_state_dict

# birefnet = BiRefNet(bb_pretrained=False)
# state_dict = torch.load('../BiRefNet-general-epoch_244.pth', map_location='cpu', weights_only=True)
# state_dict = check_state_dict(state_dict)
# birefnet.load_state_dict(state_dict)

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_float32_matmul_precision(["high", "highest"][0])

birefnet.to(device)
birefnet.eval()
print("BiRefNet is ready to use.")

# Input Data
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


src_dir = "./data/test/images"
image_paths = glob(os.path.join(src_dir, "*"))
dst_dir = "./data/test/results"
os.makedirs(dst_dir, exist_ok=True)
for image_path in image_paths:
    print("Processing {} ...".format(image_path))
    image = Image.open(image_path)
    input_images = transform_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()

    pred_pil = transforms.ToPILImage()(pred)
    pred_pil.resize(image.size).save(image_path.replace(src_dir, dst_dir))
