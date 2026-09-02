import re

with open(r'frontend\src\App.jsx', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update the header
content = re.sub(
    r"background: '#000000',\s*borderBottom: '1px solid #1e293b'",
    "background: '#ffffff',\n        borderBottom: '1px solid #e5e7eb'",
    content
)

# Fix header text colors
content = content.replace("color: '#ffffff'", "color: '#000000'")
content = content.replace("color: '#cbd5e1'", "color: '#6b7280'")
content = content.replace("background: 'rgba(255, 255, 255, 0.08)'", "background: '#f3f4f6'")
content = content.replace("color: 'rgba(255, 255, 255, 0.2)'", "color: '#d1d5db'")
content = content.replace("border: '1px solid #1e293b'", "border: '1px solid #e5e7eb'")

# 2. Replace all hard industrial borders
content = content.replace("border: '1.5px solid #000000'", "border: '1px solid #e5e7eb'")
content = content.replace("border: '1px solid #000000'", "border: '1px solid #e5e7eb'")
content = content.replace("borderBottom: '1.5px solid #000000'", "borderBottom: '1px solid #e5e7eb'")
content = content.replace("borderBottom: '1px solid #000000'", "borderBottom: '1px solid #e5e7eb'")
content = content.replace("borderRight: '1.5px solid #000000'", "borderRight: '1px solid #e5e7eb'")
content = content.replace("borderTop: '1px dashed #000000'", "borderTop: '1px dashed #e5e7eb'")

# 3. Soften the box shadows on glass panels
content = content.replace("boxShadow: '0 1px 3px rgba(0,0,0,0.02)'", "boxShadow: '0 4px 6px rgba(0,0,0,0.02)'")

# 4. Soften some grey backgrounds
content = content.replace("background: 'var(--status-slate-bg)'", "background: '#f3f4f6'")
content = content.replace("background: '#f8fafc'", "background: '#ffffff'")

# 5. Fix the background of the main container
content = content.replace("background: 'var(--bg-primary)'", "background: '#f9fafb'")

# Fix up the name spelling from earlier
content = content.replace("INSPEACT AI", "INSPECT AI")

with open(r'frontend\src\App.jsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("Refactored App.jsx")
