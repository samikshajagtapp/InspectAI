"""
Training Script for Pharmaceutical Bottle Anomaly Detection.

Uses PatchCore model architecture and Anomalib Engine to build a normal feature memory bank
from healthy bottle training images (mvtec_dataset/bottle/train/good/).
"""

import argparse
import os
# Disable artificial MPS memory allocation high watermark cap on Apple Silicon macOS
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import sys
from pathlib import Path

# Add project root to path for execution from any directory
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

# Import project utilities and configurations
from src.config import (
    DATASET_ROOT,
    CATEGORY,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    IMAGE_SIZE,
    NUM_WORKERS,
    BACKBONE,
    FEATURE_LAYERS,
    PRETRAINED,
    CORESET_SAMPLING_RATIO,
    NUM_NEIGHBORS,
    RANDOM_SEED,
    MODEL_DIRECTORY,
    CHECKPOINT_PATH,
)
from src.utils import (
    set_seed,
    ensure_directories,
    validate_dataset_structure,
    get_best_accelerator,
)


def train_patchcore(force_cpu: bool = False):
    """
    Configure and execute PatchCore model training on normal bottle images.
    """
    print("=" * 60)
    print(" PHARMACEUTICAL BOTTLE ANOMALY DETECTION - TRAINING ")
    print("=" * 60)

    # 1. Setup seed and output directories
    set_seed(RANDOM_SEED)
    ensure_directories()

    # 2. Validate dataset layout before loading
    print("\n[STEP 1/5] Validating dataset layout...")
    dataset_summary = validate_dataset_structure()
    if dataset_summary["train"].get("good", 0) == 0:
        raise RuntimeError("No training images found in mvtec_dataset/bottle/train/good/!")

    # 3. Detect and configure hardware accelerator
    print("\n[STEP 2/5] Selecting hardware accelerator...")
    accelerator, device = get_best_accelerator(force_cpu=force_cpu)

    # 4. Import Anomalib components dynamically for API resilience
    print("\n[STEP 3/5] Initializing Anomalib Datamodule & PatchCore Model...")
    
    # Imports for Datamodule
    try:
        from anomalib.data import MVTecAD
        datamodule = MVTecAD(
            root=DATASET_ROOT,
            category=CATEGORY,
            train_batch_size=TRAIN_BATCH_SIZE,
            eval_batch_size=EVAL_BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
    except (ImportError, TypeError):
        try:
            from anomalib.data import MVTec
            datamodule = MVTec(
                root=DATASET_ROOT,
                category=CATEGORY,
                train_batch_size=TRAIN_BATCH_SIZE,
                eval_batch_size=EVAL_BATCH_SIZE,
                num_workers=NUM_WORKERS,
            )
        except Exception as e:
            raise ImportError(f"Could not initialize MVTec Datamodule from anomalib.data: {e}")

    # Imports for PatchCore Model
    try:
        from anomalib.models import Patchcore
        model = Patchcore(
            backbone=BACKBONE,
            layers=FEATURE_LAYERS,
            coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
            num_neighbors=NUM_NEIGHBORS,
        )
    except (ImportError, AttributeError, TypeError):
        try:
            from anomalib.models import Patchcore
            model = Patchcore(
                backbone=BACKBONE,
                layers=FEATURE_LAYERS,
            )
        except Exception as e:
            raise ImportError(f"Could not initialize PatchCore model: {e}")

    # 5. Create Anomalib Engine / Trainer
    print("\n[STEP 4/5] Initializing Engine & Accelerator...")
    def create_engine(acc_name: str):
        try:
            from anomalib.engine import Engine
            return Engine(
                accelerator=acc_name,
                devices=1 if acc_name in ["gpu", "mps", "cuda"] else "auto",
                default_root_dir=str(MODEL_DIRECTORY),
            )
        except (ImportError, AttributeError):
            from lightning.pytorch import Trainer
            return Trainer(
                accelerator=acc_name,
                devices=1,
                default_root_dir=str(MODEL_DIRECTORY),
            )

    engine = create_engine(accelerator)

    # 6. Fit / Train the model (PatchCore extracts features & builds memory bank)
    print("\n[STEP 5/5] Fitting PatchCore memory bank on normal bottle images...")
    try:
        engine.fit(model=model, datamodule=datamodule)
    except Exception as err:
        err_msg = str(err)
        if "MPS" in err_msg or "out of memory" in err_msg.lower() or "AcceleratorError" in err_msg or "bounds" in err_msg:
            print(f"\n[WARNING] Hardware acceleration issue encountered: {err_msg}")
            print("[INFO] Automatically switching to CPU fallback for training memory bank...")
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            
            # Re-initialize engine with CPU accelerator
            engine = create_engine("cpu")
            engine.fit(model=model, datamodule=datamodule)
        else:
            raise err

    # 7. Save model checkpoint
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Engine fit standard path save
    checkpoint_to_save = CHECKPOINT_PATH
    if hasattr(engine, "save_checkpoint"):
        engine.save_checkpoint(checkpoint_to_save)
    else:
        torch.save({"state_dict": model.state_dict()}, checkpoint_to_save)

    print("\n" + "=" * 60)
    print("Training completed successfully.")
    print(f"Trained model checkpoint saved at: {checkpoint_to_save.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PatchCore Bottle Anomaly Detection Model")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training mode")
    args = parser.parse_args()

    train_patchcore(force_cpu=args.cpu)
