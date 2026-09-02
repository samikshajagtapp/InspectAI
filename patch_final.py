import re

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Add anomalib.PrecisionType to safe globals and use load_from_checkpoint
load_model_code = '''
    try:
        import anomalib
        torch.serialization.add_safe_globals([anomalib.PrecisionType])
        from anomalib.models import Patchcore
        model_instance = Patchcore.load_from_checkpoint(str(CHECKPOINT_PATH))
'''
code = re.sub(r'    try:\n        from anomalib.models import Patchcore\n        model_instance = Patchcore\(backbone=BACKBONE, layers=FEATURE_LAYERS\)\n        ckpt = torch\.load\(CHECKPOINT_PATH, map_location=\"cpu\", weights_only=False\)\n        state_dict = ckpt\.get\(\"state_dict\", ckpt\)\n        model_instance\.load_state_dict\(state_dict, strict=False\)', load_model_code, code)

# 2. Add random.shuffle to get_samples
code = code.replace('return jsonify({\"samples\": samples})', 'import random\n    random.seed(42)\n    random.shuffle(samples)\n    return jsonify({\"samples\": samples})')

# 3. Add fast mode support to predict
fast_code = '''
    fast_mode = request.form.get("fast", "false").lower() == "true"
    result = run_inference_on_cv2_image(raw_bgr, thresh_param)
    if fast_mode and "images" in result:
        del result["images"]
    return jsonify(result)
'''
code = re.sub(r'    return jsonify\(run_inference_on_cv2_image\(raw_bgr, thresh_param\)\)', fast_code, code)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
