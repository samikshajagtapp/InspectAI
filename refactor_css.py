import re

with open(r'frontend\src\index.css', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("--border-color: #000000;", "--border-color: #e5e7eb;")
content = content.replace("--border-hover: #000000;", "--border-hover: #d1d5db;")
content = content.replace("--bg-primary: #f8fafc;", "--bg-primary: #f9fafb;")

with open(r'frontend\src\index.css', 'w', encoding='utf-8') as f:
    f.write(content)

print("Refactored index.css")
