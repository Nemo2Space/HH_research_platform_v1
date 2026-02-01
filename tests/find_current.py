with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find where 'Current %' or 'Current Value' is set to '-'
for i, line in enumerate(lines):
    if 'Current %' in line or 'Current Value' in line or "'Current'" in line:
        print(f"Line {i+1}: {line[:120]}")
