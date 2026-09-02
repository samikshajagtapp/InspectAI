import sys

with open(r'original_App.jsx', 'r', encoding='utf-16') as f:
    lines = f.readlines()

in_home_block = False
home_block = []
for i, line in enumerate(lines):
    if "if (viewState === 'home')" in line or 'if (viewState === "home")' in line:
        in_home_block = True
    
    if in_home_block:
        home_block.append(line)
        # simplistic check for end of block
        if "return" in line and "<div" in lines[i+1]:
            pass
        
print("".join(home_block[:30]))
