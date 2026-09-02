from anomalib.engine import Engine
from anomalib.models import Patchcore
import torch
import anomalib
from pathlib import Path

torch.serialization.add_safe_globals([anomalib.PrecisionType])
model = Patchcore.load_from_checkpoint('models/patchcore_bottle.ckpt')

engine = Engine()
good_scores = []
broken_scores = []

for img_path in Path('mvtec_dataset/bottle/test/good').glob('*.png'):
    pred = engine.predict(model=model, data_path=str(img_path))[0]
    good_scores.append(pred.pred_score.item())

for img_path in Path('mvtec_dataset/bottle/test/broken_large').glob('*.png'):
    pred = engine.predict(model=model, data_path=str(img_path))[0]
    broken_scores.append(pred.pred_score.item())

print(f"Good normalized: min={min(good_scores)}, max={max(good_scores)}, avg={sum(good_scores)/len(good_scores)}")
print(f"Broken normalized: min={min(broken_scores)}, max={max(broken_scores)}, avg={sum(broken_scores)/len(broken_scores)}")
