import re

with open(r'frontend\src\index.css', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace brand blue with pitch black
content = content.replace("--brand-blue: #0056cc;", "--brand-blue: #000000;")
content = content.replace("--brand-blue-hover: #003d99;", "--brand-blue-hover: #333333;")
content = content.replace("--brand-blue-light: #f1f5f9;", "--brand-blue-light: #f3f4f6;")
content = content.replace("rgba(0, 86, 204, 0.02)", "rgba(0, 0, 0, 0.03)")

# Update border radius for glass panels to make it rounder (Revolut style)
content = content.replace("border-radius: 4px;", "border-radius: 12px;")

with open(r'frontend\src\index.css', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated index.css to monochrome palette")
