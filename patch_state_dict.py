import re
with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace(
'''
        if "post_processor._image_threshold" in state_dict:
            IMAGE_THRESHOLD = float(state_dict["post_processor._image_threshold"].item())
        if "post_processor._pixel_threshold" in state_dict:
            PIXEL_THRESHOLD = float(state_dict["post_processor._pixel_threshold"].item())
''',
'''
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        if "post_processor._image_threshold" in state_dict:
            IMAGE_THRESHOLD = float(state_dict["post_processor._image_threshold"].item())
        if "post_processor._pixel_threshold" in state_dict:
            PIXEL_THRESHOLD = float(state_dict["post_processor._pixel_threshold"].item())
''')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
