import re

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('''    import random, cv2''', '''    import random''')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
