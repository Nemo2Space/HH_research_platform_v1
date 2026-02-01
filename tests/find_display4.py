with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

print("Searching for render_comprehensive_stock_table and render_save_portfolio_section...")
for i, line in enumerate(lines):
    if 'render_comprehensive_stock_table' in line or 'render_save_portfolio_section' in line:
        print(f"\nLine {i+1}: {line[:120]}")
        for j in range(max(0, i-3), min(len(lines), i+5)):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1}: {lines[j][:120]}")
