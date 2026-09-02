import re

with open(r'frontend\src\App.jsx', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('color="#38bdf8"', 'color="#000000"')

with open(r'frontend\src\App.jsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated inline blues to black")
