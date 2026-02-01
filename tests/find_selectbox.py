with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Look for selectbox near strategy
for i, line in enumerate(lines):
    if 'selectbox' in line.lower():
        print(f"Line {i+1}: {line[:120]}")
