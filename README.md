# Pharmaceutical Bottle Anomaly Detection

A production-grade, deep-learning-based computer vision solution to inspect pharmaceutical bottle quality using **PatchCore** and **Anomalib**.

---

## 📌 Project Overview

In pharmaceutical manufacturing, inspecting bottle integrity (detecting cracks, breaks, and surface contamination) is critical for consumer safety and quality assurance. Traditional supervised models require thousands of labeled defective images. However, defects are rare and unpredictable.

This project implements **Unsupervised Anomaly Detection** using **PatchCore**. The model learns what a **NORMAL** bottle looks like from pristine training images and automatically flags any visual deviation as an **ANOMALY / DEFECT**.

### 🔄 Project Workflow

```
Normal Bottle Images (train/good/)
          │
          ▼
 PatchCore Feature Extraction (Wide ResNet-50-2)
          │
          ▼
   Coreset Sampling (0.1 Memory Bank Reduction)
          │
          ▼
      [ Trained Model CheckpointSaved ]
          │
          ▼
 New Bottle Image Input (predict.py)
          │
          ▼
 K-Nearest Neighbors Distance Search
          │
          ▼
   Anomaly Score & Heatmap Localization
          │
          ▼
   NORMAL / DEFECTIVE Visual Output Card
```

---

## 📂 Dataset Structure

The project uses the **MVTec AD Bottle** dataset stored at `./mvtec_dataset/bottle/`.

```
pharmacy_bottle_detection/
└── mvtec_dataset/
    └── bottle/
        ├── train/
        │   └── good/                  # 209 Normal healthy bottle images
        │
        ├── test/
        │   ├── good/                  # 20 Normal test bottles
        │   ├── broken_large/          # 20 Large fracture defect bottles
        │   ├── broken_small/          # 22 Small crack defect bottles
        │   └── contamination/         # 21 Surface contamination bottles
        │
        └── ground_truth/
            ├── broken_large/          # Pixel-level binary defect masks
            ├── broken_small/          # Pixel-level binary defect masks
            └── contamination/         # Pixel-level binary defect masks
```

---

## 🧠 PatchCore Model Architecture

**PatchCore** is an industrial anomaly detection algorithm known for speed, high accuracy, and strong localization.

- **Backbone Network:** `wide_resnet50_2` (Pretrained on ImageNet)
- **Feature Extraction Layers:** `layer2` and `layer3` (captures mid-level structural and fine-grained texture features)
- **Coreset Sampling Ratio:** `0.1` (reduces memory bank size by 90% while maintaining detection accuracy)
- **K-Nearest Neighbors (`k=9`):** Calculates distance from test patch features to nearest normal memory bank patches.

---

## 💻 Installation (macOS & Cross-Platform)

Follow these step-by-step commands to set up your virtual environment:

### Step 1: Open Terminal & Navigate to Project
```bash
cd /Users/dhruv/Desktop/Projects/pharmacy_bottle_detection
```

### Step 2: Create Python Virtual Environment
```bash
python3 -m venv .venv
```

### Step 3: Activate Virtual Environment
```bash
source .venv/bin/activate
```

### Step 4: Upgrade Package Installer
```bash
python3 -m pip install --upgrade pip
```

### Step 5: Install Project Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution & Usage Guide

### 1. Dataset Exploration (Optional)
Run the Jupyter notebook to inspect category distributions and defect masks:
```bash
jupyter notebook notebooks/exploration.ipynb
```

### 2. Train the PatchCore Model
Train the normal feature memory bank on `./mvtec_dataset/bottle/train/good/`:
```bash
python3 src/train.py
```
*Outputs: Trained checkpoint saved at `./models/patchcore_bottle.ckpt`.*

### 3. Evaluate the Model
Evaluate Image & Pixel AUROC across the test set:
```bash
python3 src/evaluate.py
```
*Outputs: Summary metrics displayed in terminal and saved to `outputs/metrics/results.json`.*

### 4. Single Image Prediction
Run anomaly detection on any bottle image:
```bash
python3 src/predict.py --image mvtec_dataset/bottle/test/broken_large/000.png
```

Example Terminal Output:
```
--------------------------------
BOTTLE ANOMALY DETECTION RESULT
--------------------------------
Image: 000.png
Prediction: DEFECTIVE
Anomaly Score: 0.8412
--------------------------------
```

---

## 📊 Output Artifacts

All results are automatically organized inside `outputs/`:

```
outputs/
├── predictions/            # Timestamped prediction folders containing original, heatmap, overlay, and final visualization card
├── heatmaps/               # Standalone anomaly color heatmaps
├── visualizations/         # High-res side-by-side presentation figures
└── metrics/
    └── results.json        # Evaluation report with Image & Pixel AUROC scores
```

---

## 🛠️ Troubleshooting & FAQ

| Problem | Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Running from outside project root | Ensure you run commands from project root or `source .venv/bin/activate`. |
| `Dataset folder NOT found` | Dataset path misplaced | Verify `./mvtec_dataset/bottle/` exists with `train/good/`. |
| `Checkpoint missing` | Trying to evaluate/predict before training | Run `python3 src/train.py` first. |
| `MPS acceleration error` | Apple Silicon PyTorch build issue | `src/utils.py` automatically falls back to CPU cleanly without crashing. |
| `CUDA unavailable` | Running on Mac or CPU machine | The system automatically selects CPU or Apple MPS. No CUDA GPU required. |

---

## 📜 License & Acknowledgments
Dataset provided by [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad). Model powered by [Anomalib](https://github.com/openvinotoolkit/anomalib).
