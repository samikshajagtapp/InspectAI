import torch, cv2, numpy as np
from pathlib import Path
from torchvision import transforms
from anomalib.models import Patchcore
import anomalib

torch.serialization.add_safe_globals([anomalib.PrecisionType])
model = Patchcore.load_from_checkpoint('models/patchcore_bottle.ckpt')
model.eval()

t = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((256, 256)), 
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

for cat in ['good', 'broken_large', 'contamination']:
    p = list(Path(f'mvtec_dataset/bottle/test/{cat}').glob('*.png'))[0]
    im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        out = model.model(t(im).unsqueeze(0))
        hm = out.anomaly_map.squeeze().cpu().numpy()
        print(f"{cat}: min={hm.min():.4f}, max={hm.max():.4f}, mean={hm.mean():.4f}, std={hm.std():.4f}")
