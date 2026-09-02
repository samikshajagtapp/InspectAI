from anomalib.engine import Engine
from anomalib.models import Patchcore
import torch
import anomalib

torch.serialization.add_safe_globals([anomalib.PrecisionType])
model = Patchcore.load_from_checkpoint('models/patchcore_bottle.ckpt')

engine = Engine()
predictions = engine.predict(model=model, data_path='mvtec_dataset/bottle/test/good/000.png')
print("Engine Prediction:", predictions)
