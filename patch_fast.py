import re

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

new_code = '''
        result = run_inference_on_cv2_image(raw_bgr, threshold=threshold)
        if request.form.get("fast", "false").lower() == "true" and "images" in result:
            del result["images"]
        return jsonify(result)
'''
code = code.replace('result = run_inference_on_cv2_image(raw_bgr, threshold=threshold)\n        return jsonify(result)', new_code.strip())

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
