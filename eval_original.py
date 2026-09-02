import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms

from src.config import BACKBONE, FEATURE_LAYERS, IMAGE_SIZE
from anomalib.models import Patchcore

model_instance = Patchcore(backbone=BACKBONE, layers=FEATURE_LAYERS)
ckpt = torch.load('models/patchcore_bottle.ckpt', map_location='cpu', weights_only=False)
state_dict = ckpt.get('state_dict', ckpt)
model_instance.load_state_dict(state_dict, strict=False)
model_instance.eval()
model_instance.to('cpu')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for cat in ['good', 'broken_large']:
    img_path = list((Path('mvtec_dataset/bottle/test') / cat).glob('*.png'))[0]
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor_img = transform(img_rgb).unsqueeze(0)
    
    with torch.no_grad():
        out = model_instance.model(tensor_img)
        print(f"{cat} ORIGINAL RAW:", float(out.pred_score.item()))
