with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find where the BUY/SELL display is rendered with HTML
for i, line in enumerate(lines):
    if 'span' in line.lower() and ('buy' in line.lower() or 'sell' in line.lower() or 'green' in line.lower() or 'red' in line.lower()):
        print(f"Line {i+1}: {line[:120]}")
