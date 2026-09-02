import torch
import cv2
from pathlib import Path
from torchvision import transforms

import anomalib
torch.serialization.add_safe_globals([anomalib.PrecisionType])
from anomalib.models import Patchcore

model = Patchcore.load_from_checkpoint('models/patchcore_bottle.ckpt')
model.eval()
model.to('cpu')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for cat in ['good', 'broken_large']:
    img_path = list((Path('mvtec_dataset/bottle/test') / cat).glob('*.png'))[0]
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor_img = transform(img_rgb).unsqueeze(0)
    
    with torch.no_grad():
        out = model(tensor_img)
        print(f"{cat}:", float(out.pred_score.item()))
