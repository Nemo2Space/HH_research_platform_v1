with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find the _render_portfolio_comparison function to see current state
for i, line in enumerate(lines):
    if 'def _render_portfolio_comparison' in line:
        print(f"Lines {i+1}-{i+60}:")
        for j in range(i, min(len(lines), i+60)):
            print(f"{j+1}: {lines[j]}")
        break
