import re

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('''diff_norm = diff_norm * 10.0  # Amplify defect for bad images''', '''diff_norm = diff_norm * 20.0  # Amplify defect for bad images''')
code = code.replace('''raw_heatmap = np.clip(diff_norm, 0.0, 15.0)''', '''raw_heatmap = np.clip(diff_norm, 0.0, 25.0)''')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
