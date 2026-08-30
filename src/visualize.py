"""
Visualization Utilities for Pharmaceutical Bottle Anomaly Detection.

Generates high-resolution multi-panel plots showcasing original bottle images,
anomaly heatmaps, overlay representations, and prediction score cards.
"""

from pathlib import Path
from typing import Tuple, Union, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    Normalize an anomaly heatmap array to the range [0, 1].

    Args:
        heatmap (np.ndarray): 2D or 3D raw anomaly heatmap tensor/array.

    Returns:
        np.ndarray: Normalized 2D float heatmap in range [0, 1].
    """
    if heatmap.ndim == 3:
        heatmap = heatmap.squeeze()
    
    min_val, max_val = heatmap.min(), heatmap.max()
    if max_val - min_val > 1e-6:
        normalized = (heatmap - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(heatmap, dtype=np.float32)
    return normalized.astype(np.float32)


def generate_heatmap_overlay(
    image_rgb: np.ndarray, heatmap_norm: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """
    Blend an RGB image with a colormapped anomaly heatmap overlay.

    Args:
        image_rgb (np.ndarray): RGB uint8 image (H, W, 3).
        heatmap_norm (np.ndarray): 2D float heatmap normalized to [0, 1].
        alpha (float): Blending factor for overlay.

    Returns:
        np.ndarray: Blended RGB uint8 image overlay.
    """
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match image if dimensions differ
    if heatmap_colored_rgb.shape[:2] != image_rgb.shape[:2]:
        heatmap_colored_rgb = cv2.resize(
            heatmap_colored_rgb, (image_rgb.shape[1], image_rgb.shape[0])
        )

    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_colored_rgb, alpha, 0)
    return overlay


def add_anomaly_red_markings(
    image_rgb: np.ndarray,
    heatmap_raw: np.ndarray,
    is_defective: bool,
    pixel_threshold: float = 12.20,
    draw_box: bool = True,
    draw_red_patch: bool = True,
) -> np.ndarray:
    """
    Draw a prominent red patch fill and red bounding box markings around detected defect regions.
    Only draws markings if is_defective is True and pixels exceed pixel_threshold.

    Args:
        image_rgb (np.ndarray): Input RGB image array.
        heatmap_raw (np.ndarray): 2D raw feature-distance anomaly heatmap.
        is_defective (bool): Whether the bottle is classified as DEFECTIVE.
        pixel_threshold (float): Anomaly distance threshold per pixel.
        draw_box (bool): If True, draws red bounding boxes around anomaly patches.
        draw_red_patch (bool): If True, overlays a vivid red patch on anomalous regions.

    Returns:
        np.ndarray: Marked RGB image array.
    """
    marked_img = image_rgb.copy()
    
    # If the bottle is normal/good, return clean image without any red defect markings
    if not is_defective:
        return marked_img
    
    # Resize raw heatmap to match image dimensions
    if heatmap_raw.shape[:2] != marked_img.shape[:2]:
        heatmap_resized = cv2.resize(
            heatmap_raw, (marked_img.shape[1], marked_img.shape[0])
        )
    else:
        heatmap_resized = heatmap_raw.copy()
    
    # Isolate pixels that exceed raw anomaly pixel threshold
    mask = (heatmap_resized >= pixel_threshold).astype(np.uint8) * 255
    
    # Fallback to top 5% if strict pixel threshold is slightly missed but bottle is defective
    if np.sum(mask) == 0:
        p95 = float(np.percentile(heatmap_resized, 95))
        mask = (heatmap_resized >= p95).astype(np.uint8) * 255

    # 1. Apply vivid red semi-transparent patch fill over true anomaly region
    if draw_red_patch and np.sum(mask) > 0:
        red_layer = np.zeros_like(marked_img, dtype=np.uint8)
        red_layer[:, :] = [255, 0, 0]  # Red in RGB
        
        patch_mask = (mask > 0)[:, :, None]
        blended = cv2.addWeighted(marked_img, 0.4, red_layer, 0.6, 0)
        marked_img = np.where(patch_mask, blended, marked_img)

    # 2. Find contours and draw red outline + bounding box
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 15:  # Filter noise dots
            # Contour border in red
            cv2.drawContours(marked_img, [cnt], -1, (255, 0, 0), 2)
            
            if draw_box:
                x, y, w, h = cv2.boundingRect(cnt)
                pad = 4
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(marked_img.shape[1], x + w + pad)
                y2 = min(marked_img.shape[0], y + h + pad)
                
                # Draw thick red bounding rectangle
                cv2.rectangle(marked_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                
                # Draw badge header label
                label = "DEFECT DETECTED"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)
                tag_y1 = max(0, y1 - th - 6)
                cv2.rectangle(marked_img, (x1, tag_y1), (x1 + tw + 8, y1), (255, 0, 0), -1)
                cv2.putText(
                    marked_img,
                    label,
                    (x1 + 4, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    return marked_img


def create_side_by_side_visualization(
    image_rgb: np.ndarray,
    heatmap_norm: np.ndarray,
    is_defective: bool,
    anomaly_score: float,
    output_path: Optional[Union[str, Path]] = None,
    threshold: float = 10.46,
    heatmap_raw: Optional[np.ndarray] = None,
    pixel_threshold: float = 12.20,
) -> np.ndarray:
    """
    Create a professional 3-panel figure:
      Panel 1: Original Bottle
      Panel 2: Anomaly Heatmap
      Panel 3: Defect Red Patch & Box Marking

    Displays prediction label (NORMAL / DEFECTIVE) and score in header card.
    """
    if is_defective:
        raw_map = heatmap_raw if heatmap_raw is not None else heatmap_norm
        marked_rgb = add_anomaly_red_markings(
            image_rgb, raw_map, is_defective=is_defective, pixel_threshold=pixel_threshold
        )
        panel_title = "Defect Red Patch & Marking"
    else:
        marked_rgb = image_rgb.copy()
        panel_title = "Clean Bottle (No Anomaly)"

    # Set up matplotlib style figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=300)
    fig.patch.set_facecolor("#1e1e2e")  # Sleek dark background

    for ax in axes:
        ax.set_facecolor("#1e1e2e")
        ax.axis("off")

    # 1. Original Image
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Bottle", color="#ffffff", fontsize=14, pad=10, fontweight="bold")

    # 2. Anomaly Heatmap
    im = axes[1].imshow(heatmap_norm, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Anomaly Heatmap", color="#ffffff", fontsize=14, pad=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#ffffff")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ffffff")

    # 3. Anomaly Red Patch & Marking
    axes[2].imshow(marked_rgb)
    axes[2].set_title(panel_title, color="#ffffff", fontsize=14, pad=10, fontweight="bold")

    # Header Prediction Badge & Title
    status_str = "DEFECTIVE" if is_defective else "NORMAL"
    badge_color = "#ff4949" if is_defective else "#20bf6b"

    title_text = f"Prediction: {status_str} | Anomaly Score: {anomaly_score:.4f}"
    fig.suptitle(
        title_text,
        color=badge_color,
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        print(f"[INFO] Visualization saved to: {out_p.resolve()}")

    # Convert plot figure to RGB array if needed for downstream saving
    fig.canvas.draw()
    try:
        plot_img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    except AttributeError:
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return plot_img
