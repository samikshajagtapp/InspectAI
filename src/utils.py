"""
Utility Functions for Pharmaceutical Bottle Anomaly Detection.

Provides random seed setting, directory creation, dataset structure validation,
and robust hardware accelerator selection (CUDA, Apple MPS, or CPU).
"""

import os
# Disable artificial MPS memory allocation high watermark cap on Apple Silicon macOS
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import random
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch

from src.config import (
    PROJECT_ROOT,
    DATASET_ROOT,
    CATEGORY,
    BOTTLE_DATASET_PATH,
    TRAIN_GOOD_PATH,
    TEST_PATH,
    GROUND_TRUTH_PATH,
    MODEL_DIRECTORY,
    OUTPUT_DIRECTORY,
    PREDICTIONS_DIRECTORY,
    HEATMAPS_DIRECTORY,
    VISUALIZATIONS_DIRECTORY,
    METRICS_DIRECTORY,
    RANDOM_SEED,
)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed (int): The random seed value. Defaults to RANDOM_SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed set to: {seed}")


def ensure_directories() -> None:
    """
    Create all required project directories if they do not exist.
    """
    dirs = [
        MODEL_DIRECTORY,
        OUTPUT_DIRECTORY,
        PREDICTIONS_DIRECTORY,
        HEATMAPS_DIRECTORY,
        VISUALIZATIONS_DIRECTORY,
        METRICS_DIRECTORY,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] All output directories verified under: {OUTPUT_DIRECTORY}")


def check_dataset_exists(dataset_root: Path = DATASET_ROOT, category: str = CATEGORY) -> bool:
    """
    Check if the specified MVTec AD dataset directory exists.

    Args:
        dataset_root (Path): Root directory of dataset.
        category (str): Category subfolder name.

    Returns:
        bool: True if dataset directory exists, False otherwise.
    """
    target_path = dataset_root / category
    exists = target_path.exists() and target_path.is_dir()
    if exists:
        print(f"[INFO] Dataset folder found at: {target_path}")
    else:
        print(f"[WARNING] Dataset folder NOT found at: {target_path}")
    return exists


def validate_dataset_structure() -> Dict[str, Any]:
    """
    Validate the internal folder structure of the MVTec AD bottle dataset
    and count images in each directory.

    Returns:
        Dict[str, Any]: Detailed summary dictionary of image counts.
    """
    if not check_dataset_exists():
        raise FileNotFoundError(
            f"Dataset missing at {BOTTLE_DATASET_PATH}. Please check dataset path."
        )

    summary = {"train": {}, "test": {}, "ground_truth": {}}

    # Validate train/good
    if TRAIN_GOOD_PATH.exists():
        train_imgs = list(TRAIN_GOOD_PATH.glob("*.png")) + list(TRAIN_GOOD_PATH.glob("*.jpg"))
        summary["train"]["good"] = len(train_imgs)
    else:
        print(f"[WARNING] Missing directory: {TRAIN_GOOD_PATH}")
        summary["train"]["good"] = 0

    # Validate test subdirectories
    if TEST_PATH.exists():
        for sub_dir in TEST_PATH.iterdir():
            if sub_dir.is_dir():
                test_imgs = list(sub_dir.glob("*.png")) + list(sub_dir.glob("*.jpg"))
                summary["test"][sub_dir.name] = len(test_imgs)

    # Validate ground_truth subdirectories
    if GROUND_TRUTH_PATH.exists():
        for sub_dir in GROUND_TRUTH_PATH.iterdir():
            if sub_dir.is_dir():
                gt_imgs = list(sub_dir.glob("*.png")) + list(sub_dir.glob("*.jpg"))
                summary["ground_truth"][sub_dir.name] = len(gt_imgs)

    print("[INFO] Dataset Structure Validation:")
    print(f"  - Training Normal Images (train/good): {summary['train'].get('good', 0)}")
    print("  - Test Set Image Counts:")
    for cat_name, count in summary["test"].items():
        print(f"      * {cat_name}: {count}")
    print("  - Ground Truth Mask Counts:")
    for cat_name, count in summary["ground_truth"].items():
        print(f"      * {cat_name}: {count}")

    return summary


def get_best_accelerator(force_cpu: bool = False) -> Tuple[str, str]:
    """
    Select the optimal hardware accelerator available on the host machine.
    Hierarchy:
      1. CPU if requested explicitly or via FORCE_CPU env var
      2. CUDA (NVIDIA GPU)
      3. MPS (Apple Silicon GPU) if supported
      4. CPU (Fallback)

    Returns:
        Tuple[str, str]: (accelerator_name, device_string)
        e.g., ("gpu", "cuda"), ("mps", "mps"), or ("cpu", "cpu")
    """
    if force_cpu or os.environ.get("FORCE_CPU", "0") in ("1", "true", "TRUE"):
        print("[INFO] Hardware Accelerator: CPU forced by user configuration.")
        return "cpu", "cpu"

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[INFO] Hardware Accelerator: NVIDIA CUDA GPU detected ({device_name})")
        return "gpu", "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            print("[INFO] Hardware Accelerator: Apple Silicon MPS detected & enabled")
            return "mps", "mps"
        else:
            print("[WARNING] Apple MPS is available but PyTorch was not built with MPS support. Falling back to CPU.")
            return "cpu", "cpu"
    else:
        print("[INFO] Hardware Accelerator: No CUDA or MPS detected. Using CPU")
        return "cpu", "cpu"
