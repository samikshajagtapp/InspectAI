"""
Central Configuration for Pharmaceutical Bottle Anomaly Detection.

This file defines all system parameters, directory paths, model hyperparameters,
and hardware acceleration defaults using pathlib for cross-platform robustness.
"""

import sys
from pathlib import Path

# ==========================================
# 1. Project & Path Configurations
# ==========================================
# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset root directory and category configuration
DATASET_ROOT = PROJECT_ROOT / "mvtec_dataset"
CATEGORY = "bottle"

# Specific paths within dataset for validation
BOTTLE_DATASET_PATH = DATASET_ROOT / CATEGORY
TRAIN_GOOD_PATH = BOTTLE_DATASET_PATH / "train" / "good"
TEST_PATH = BOTTLE_DATASET_PATH / "test"
GROUND_TRUTH_PATH = BOTTLE_DATASET_PATH / "ground_truth"

# Model artifact output directory
MODEL_DIRECTORY = PROJECT_ROOT / "models"

# Results and visualization output directories
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
PREDICTIONS_DIRECTORY = OUTPUT_DIRECTORY / "predictions"
HEATMAPS_DIRECTORY = OUTPUT_DIRECTORY / "heatmaps"
VISUALIZATIONS_DIRECTORY = OUTPUT_DIRECTORY / "visualizations"
METRICS_DIRECTORY = OUTPUT_DIRECTORY / "metrics"

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
IMAGE_SIZE = (256, 256)
NUM_WORKERS = 2 if sys.platform != "win32" else 0
RANDOM_SEED = 42

# ==========================================
# 3. PatchCore Model Hyperparameters
# ==========================================
MODEL_NAME = "patchcore"
BACKBONE = "resnet18"
FEATURE_LAYERS = ["layer2", "layer3"]
PRETRAINED = True
CORESET_SAMPLING_RATIO = 0.01
NUM_NEIGHBORS = 9

# Default checkpoint name
CHECKPOINT_NAME = "patchcore_bottle.ckpt"
CHECKPOINT_PATH = MODEL_DIRECTORY / CHECKPOINT_NAME
