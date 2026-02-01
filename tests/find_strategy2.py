with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    if 'Strategy' in line and 'st.' in line:
        print(f"Line {i+1}: {line[:120]}")
