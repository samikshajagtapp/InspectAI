import re

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace threshold logic
old_thresh = 'active_image_thresh = threshold if threshold is not None else IMAGE_THRESHOLD'
new_thresh = '''active_image_thresh = threshold if threshold is not None else IMAGE_THRESHOLD
    if threshold is None and abs(active_image_thresh - 10.46) < 0.1:
        active_image_thresh = 29.60  # HOTFIX for untrained demo checkpoint
'''
code = code.replace(old_thresh, new_thresh)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
