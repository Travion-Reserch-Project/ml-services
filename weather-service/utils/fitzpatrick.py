import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import torch
import timm

IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

def preprocess_image(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    image = np.array(image)
    image = transform(image=image)["image"]
    return image.unsqueeze(0)

def load_model():
    model = timm.create_model('efficientnetv2_rw_m', pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, 6)

    model.load_state_dict(
        torch.load("models/efficientnetv2_fitzpatrick_best.pth", map_location=device)
    )

    model.to(device)
    model.eval()
    return model