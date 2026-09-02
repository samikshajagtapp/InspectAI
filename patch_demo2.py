with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'result["filename"] = file.filename' in line:
        lines[i-1] = lines[i-1].replace('sample_path=sample_path_rel or ""', 'sample_path=file.filename')

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
