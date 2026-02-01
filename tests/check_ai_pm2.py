with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find the main render function and strategy selection
print("Searching for render function and strategy selection...")
for i, line in enumerate(lines):
    if 'def render_ai_portfolio_manager_tab' in line or 'Strategy' in line and 'select' in line.lower():
        print(f"\nLine {i+1}: {line[:100]}")
        for j in range(max(0, i-2), min(len(lines), i+15)):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1}: {lines[j][:100]}")
