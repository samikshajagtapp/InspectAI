"""
Evaluation Script for Pharmaceutical Bottle Anomaly Detection.

Evaluates trained PatchCore model on MVTec AD bottle test dataset (good, broken_large,
broken_small, contamination) and records Image & Pixel AUROC, F1 score, Precision, and Recall.
Saves metrics report to outputs/metrics/results.json.
"""

import json
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Auto-switch to project virtual environment if dependencies are missing in system python
try:
    import anomalib
except ImportError:
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python3"
    if venv_python.exists() and sys.executable != str(venv_python):
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

import torch

from src.config import (
    DATASET_ROOT,
    CATEGORY,
    EVAL_BATCH_SIZE,
    IMAGE_SIZE,
    NUM_WORKERS,
    BACKBONE,
    FEATURE_LAYERS,
    CHECKPOINT_PATH,
    METRICS_DIRECTORY,
)
from src.utils import ensure_directories, get_best_accelerator


def evaluate_model(checkpoint_path: Path = CHECKPOINT_PATH):
    """
    Evaluate the trained PatchCore model against the test set and ground truth masks.

    Args:
        checkpoint_path (Path): Path to model checkpoint file.
    """
    print("=" * 60)
    print(" PHARMACEUTICAL BOTTLE ANOMALY DETECTION - EVALUATION ")
    print("=" * 60)

    ensure_directories()

    # Check model checkpoint existence
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found at: {checkpoint_path}")
        print("Please run 'python3 src/train.py' first to train the model.")
        sys.exit(1)

    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")

    # Select hardware accelerator
    accelerator, device = get_best_accelerator()

    # Import anomalib datamodule & engine
    try:
        from anomalib.data import MVTecAD
        datamodule = MVTecAD(
            root=DATASET_ROOT,
            category=CATEGORY,
            eval_batch_size=EVAL_BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
    except (ImportError, TypeError):
        from anomalib.data import MVTec
        datamodule = MVTec(
            root=DATASET_ROOT,
            category=CATEGORY,
            eval_batch_size=EVAL_BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

    # Initialize model
    try:
        from anomalib.models import Patchcore
        model = Patchcore.load_from_checkpoint(str(checkpoint_path))
    except Exception as e:
        print(f"[INFO] Standard load_from_checkpoint failed ({e}), creating model instance...")
        from anomalib.models import Patchcore
        try:
            model = Patchcore(backbone=BACKBONE, layers=FEATURE_LAYERS)
        except Exception:
            model = Patchcore(backbone=BACKBONE)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    # Create Engine
    try:
        from anomalib.engine import Engine
        engine = Engine(accelerator=accelerator, devices=1 if accelerator in ["gpu", "mps", "cuda"] else "auto")
    except (ImportError, AttributeError):
        from lightning.pytorch import Trainer
        engine = Trainer(accelerator=accelerator, devices=1)

    print("[INFO] Running evaluation on test set...")
    test_results = engine.test(model=model, datamodule=datamodule)

    # Extract metrics safely
    metrics_summary = {}
    if test_results and isinstance(test_results, list) and len(test_results) > 0:
        res_dict = test_results[0]
        for key, val in res_dict.items():
            if isinstance(val, (int, float)):
                metrics_summary[key] = round(float(val), 4)
            elif torch.is_tensor(val):
                metrics_summary[key] = round(float(val.item()), 4)

    # Key metric fallback formatting
    extracted_results = {
        "dataset_category": CATEGORY,
        "image_AUROC": metrics_summary.get("image_AUROC", metrics_summary.get("image_rocauc", "N/A")),
        "pixel_AUROC": metrics_summary.get("pixel_AUROC", metrics_summary.get("pixel_rocauc", "N/A")),
        "image_F1Score": metrics_summary.get("image_F1Score", metrics_summary.get("image_f1", "N/A")),
        "pixel_F1Score": metrics_summary.get("pixel_F1Score", metrics_summary.get("pixel_f1", "N/A")),
        "image_Precision": metrics_summary.get("image_Precision", metrics_summary.get("image_precision", "N/A")),
        "image_Recall": metrics_summary.get("image_Recall", metrics_summary.get("image_recall", "N/A")),
        "raw_metrics": metrics_summary,
    }

    # Save metrics JSON
    METRICS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    results_path = METRICS_DIRECTORY / "results.json"
    with open(results_path, "w") as f:
        json.dump(extracted_results, f, indent=4)

    print("\n" + "=" * 60)
    print(" EVALUATION RESULTS ")
    print("=" * 60)
    print(f" Category         : {extracted_results['dataset_category']}")
    print(f" Image AUROC      : {extracted_results['image_AUROC']}")
    print(f" Pixel AUROC      : {extracted_results['pixel_AUROC']}")
    print(f" Image F1 Score   : {extracted_results['image_F1Score']}")
    print(f" Image Precision  : {extracted_results['image_Precision']}")
    print(f" Image Recall     : {extracted_results['image_Recall']}")
    print("-" * 60)
    print(f"[SUCCESS] Metrics saved to: {results_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()
