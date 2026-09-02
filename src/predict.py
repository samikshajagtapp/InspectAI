"""
Single-Image Prediction Script for Pharmaceutical Bottle Anomaly Detection.

Accepts an input image path via command-line interface, runs trained PatchCore inference,
determines NORMAL vs DEFECTIVE status, computes the anomaly score, and outputs visual artifacts.

Usage:
  python3 src/predict.py --image path/to/bottle_image.png
"""

import argparse
from datetime import datetime
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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config import (
    IMAGE_SIZE,
    BACKBONE,
    FEATURE_LAYERS,
    CHECKPOINT_PATH,
    PREDICTIONS_DIRECTORY,
    HEATMAPS_DIRECTORY,
    VISUALIZATIONS_DIRECTORY,
)
from src.utils import ensure_directories, get_best_accelerator
from src.visualize import (
    normalize_heatmap,
    generate_heatmap_overlay,
    add_anomaly_red_markings,
    create_side_by_side_visualization,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PatchCore Anomaly Detection on a single bottle image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input bottle image file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_PATH),
        help="Path to trained model checkpoint file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Anomaly score threshold to classify image as DEFECTIVE (default: 0.5).",
    )
    return parser.parse_args()


def predict_single_image(image_path_str: str, checkpoint_path_str: str, threshold: float = 0.5):
    """
    Execute single image anomaly inference.
    """
    image_path = Path(image_path_str).resolve()
    checkpoint_path = Path(checkpoint_path_str).resolve()

    # 1. Validate image path
    if not image_path.exists():
        print(f"[ERROR] Specified image file does not exist: {image_path}")
        sys.exit(1)

    # 2. Validate checkpoint
    if not checkpoint_path.exists():
        print(f"[ERROR] Trained model checkpoint missing at: {checkpoint_path}")
        print("Please train the PatchCore model first using 'python3 src/train.py'.")
        sys.exit(1)

    ensure_directories()
    accelerator, device_str = get_best_accelerator()

    # 3. Read image
    raw_bgr = cv2.imread(str(image_path))
    if raw_bgr is None:
        print(f"[ERROR] Failed to read image with OpenCV: {image_path}")
        sys.exit(1)
    
    image_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

    # 4. Load trained model
    print(f"[INFO] Loading PatchCore model checkpoint from: {checkpoint_path}")
    try:
        from anomalib.models import Patchcore
        model = Patchcore.load_from_checkpoint(str(checkpoint_path))
    except Exception as e:
        print(f"[INFO] Standard load_from_checkpoint fallback ({e}), creating model shell...")
        from anomalib.models import Patchcore
        try:
            model = Patchcore(backbone=BACKBONE, layers=FEATURE_LAYERS)
        except Exception:
            model = Patchcore(backbone=BACKBONE)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    model.to(device_str)
    model.eval()

    # Preprocess image tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor_img = transform(image_rgb).unsqueeze(0).to(device_str)

    # 5. Run inference
    with torch.no_grad():
        try:
            outputs = model(tensor_img)
        except Exception:
            try:
                outputs = model.predict_step({"image": tensor_img}, batch_idx=0)
            except Exception:
                outputs = model.predict_step(tensor_img, batch_idx=0)

        if isinstance(outputs, dict):
            anomaly_map = outputs.get("anomaly_map", outputs.get("anomaly_maps", None))
            pred_score_val = outputs.get("pred_score", None)
        else:
            anomaly_map = getattr(outputs, "anomaly_map", getattr(outputs, "anomaly_maps", None))
            pred_score_val = getattr(outputs, "pred_score", None)

        if pred_score_val is not None:
            pred_score = float(pred_score_val.item() if hasattr(pred_score_val, "item") else pred_score_val)
        elif anomaly_map is not None:
            pred_score = float(anomaly_map.max().item())
        else:
            pred_score = 0.5

    # Process heatmap map tensor
    if anomaly_map is not None:
        if torch.is_tensor(anomaly_map):
            heatmap_np = anomaly_map.squeeze().cpu().numpy()
        else:
            heatmap_np = np.array(anomaly_map).squeeze()
    else:
        heatmap_np = np.zeros(IMAGE_SIZE, dtype=np.float32)

    heatmap_norm = normalize_heatmap(heatmap_np)
    is_defective = pred_score >= threshold
    status_text = "DEFECTIVE" if is_defective else "NORMAL"

    # 6. Terminal Summary Report
    print("\n" + "-" * 40)
    print("BOTTLE ANOMALY DETECTION RESULT")
    print("-" * 40)
    print(f"Image: {image_path.name}")
    print(f"Prediction: {status_text}")
    print(f"Anomaly Score: {pred_score:.4f}")
    print("-" * 40 + "\n")

    # 7. Create unique output directory for prediction run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_folder = PREDICTIONS_DIRECTORY / f"{image_path.stem}_{timestamp}"
    out_folder.mkdir(parents=True, exist_ok=True)

    # Save original image
    cv2.imwrite(str(out_folder / "original.png"), cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))

    # Save raw heatmap
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_folder / "heatmap.png"), heatmap_colored_bgr)

    # Save overlay
    overlay_rgb = generate_heatmap_overlay(image_resized, heatmap_norm, alpha=0.5)
    cv2.imwrite(str(out_folder / "overlay.png"), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    # Save red patch & bounding box anomaly marking image
    if is_defective:
        red_marked_rgb = add_anomaly_red_markings(image_resized, heatmap_norm, threshold=threshold)
        cv2.imwrite(str(out_folder / "red_marked_bottle.png"), cv2.cvtColor(red_marked_rgb, cv2.COLOR_RGB2BGR))

    # Save side-by-side multi-panel visualization
    create_side_by_side_visualization(
        image_rgb=image_resized,
        heatmap_norm=heatmap_norm,
        is_defective=is_defective,
        anomaly_score=pred_score,
        output_path=out_folder / "final_visualization.png",
        threshold=threshold,
    )

    print(f"[SUCCESS] All output predictions saved to: {out_folder.resolve()}")


if __name__ == "__main__":
    args = parse_args()
    predict_single_image(args.image, args.checkpoint, args.threshold)
