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
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

scores = {'good': [], 'defective': []}

for cat_dir in Path('mvtec_dataset/bottle/test').iterdir():
    cat = cat_dir.name
    key = 'good' if cat == 'good' else 'defective'
    for img_path in list(cat_dir.glob('*.png'))[:5]:
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor_img = transform(img_rgb).unsqueeze(0)
        with torch.no_grad():
            out = model.model(tensor_img)
            scores[key].append(float(out.pred_score.item()))

print('Good:', scores['good'])
print('Defective:', scores['defective'])
