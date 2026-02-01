with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find Portfolio Comparison rendering
for i, line in enumerate(lines):
    if 'Portfolio Comparison' in line or 'Current vs Target' in line:
        print(f"\nLine {i+1}: {line[:100]}")
        for j in range(max(0, i-5), min(len(lines), i+40)):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1}: {lines[j][:110]}")
        break
