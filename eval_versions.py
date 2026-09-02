import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms

from anomalib.models import Patchcore
import anomalib
torch.serialization.add_safe_globals([anomalib.PrecisionType])

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for ver in ['v6', 'v7', 'v8']:
    ckpt_path = f'models/Patchcore/MVTecAD/bottle/{ver}/weights/lightning/model.ckpt'
    model = Patchcore.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to('cpu')
    
    print(f"--- {ver} ---")
    for cat in ['good', 'broken_large']:
        img_path = list((Path('mvtec_dataset/bottle/test') / cat).glob('*.png'))[0]
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor_img = transform(img_rgb).unsqueeze(0)
        
        with torch.no_grad():
            out = model.model(tensor_img)
            print(f"{cat}:", float(out.pred_score.item()))
