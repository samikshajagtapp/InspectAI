import torch, cv2
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

def get_scores(cat):
    s = []
    for p in Path(f'mvtec_dataset/bottle/test/{cat}').glob('*.png'):
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            s.append(float(model.model(t(im).unsqueeze(0)).pred_score.item()))
    return s

g = get_scores('good')
b_l = get_scores('broken_large')
b_s = get_scores('broken_small')
c = get_scores('contamination')

print(f"Good: {min(g):.2f} - {max(g):.2f}")
print(f"Broken L: {min(b_l):.2f} - {max(b_l):.2f}")
print(f"Broken S: {min(b_s):.2f} - {max(b_s):.2f}")
print(f"Contam: {min(c):.2f} - {max(c):.2f}")
