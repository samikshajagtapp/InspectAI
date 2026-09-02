import re

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

old_demo = '''    # DEMO MODE HOTFIX: Generate realistic scores matching the manual's specifications
    import random
    if sample_path:
        path_lower = sample_path.lower()
        if "/good/" in path_lower:
            raw_score = random.uniform(2.5, 9.5) # Auto-Pass (< 10.46)
        elif "/broken_small/" in path_lower:
            raw_score = random.uniform(10.50, 13.40) # Human-Review (10.46 - 13.50)
        else:
            raw_score = random.uniform(13.60, 24.50) # Auto-Reject (> 13.50)'''

code = code.replace(old_demo, '')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
