import re
with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Add action calculation after is_defective
action_code = '''
    if not is_defective:
        action = "AUTO-PASS"
    elif raw_score < active_image_thresh * 1.15:
        action = "HUMAN-REVIEW"
    else:
        action = "AUTO-REJECT"
'''
code = code.replace('is_defective = (raw_score >= active_image_thresh)', 'is_defective = (raw_score >= active_image_thresh)\n' + action_code)

# Add action to return dict
code = code.replace('\"is_defective\": is_defective,', '\"is_defective\": is_defective,\n        \"action\": action,')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
