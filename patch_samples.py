import re

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

new_samples_code = '''@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return all sample bottle images from dataset."""
    samples = []
    test_dir = DATASET_ROOT / "bottle" / "test"
    if test_dir.exists():
        for category_dir in sorted(test_dir.iterdir()):
            if category_dir.is_dir():
                for img_path in sorted(category_dir.glob("*.png")):
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is not None:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        img_thumb = cv2.resize(img_rgb, (128, 128))
                        title = "Normal Bottle" if category_dir.name == "good" else f"{category_dir.name.replace('_', ' ').title()} Defect"
                        samples.append({
                            "id": f"{category_dir.name}_{img_path.stem}",
                            "title": title,
                            "category": category_dir.name,
                            "path": f"bottle/test/{category_dir.name}/{img_path.name}",
                            "description": f"Sample from {category_dir.name} category",
                            "image_b64": numpy_to_base64(img_thumb)
                        })
    return jsonify({"samples": samples})'''

code = re.sub(r'@app\.route\("/api/samples", methods=\["GET"\]\).*?(?=@app\.route\("/api/predict")', new_samples_code + '\n\n\n', code, flags=re.DOTALL)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
