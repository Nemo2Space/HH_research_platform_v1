with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find where render_enhanced_portfolio or the portfolio display is called
print("Searching for portfolio display/render logic...")
for i, line in enumerate(lines):
    if 'render_enhanced' in line.lower() or 'display_portfolio' in line.lower() or 'render_portfolio' in line.lower():
        print(f"\nLine {i+1}: {line[:100]}")
        for j in range(max(0, i-2), min(len(lines), i+5)):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1}: {lines[j][:100]}")
