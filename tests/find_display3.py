with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show the import and find usage
print("Lines 87-100:")
for i in range(87, 105):
    print(f"{i+1}: {lines[i]}")

print("\n\nSearching for display_enhanced or render functions from that import...")
for i, line in enumerate(lines):
    if 'display_enhanced' in line or 'EnhancedPortfolioDisplay' in line or 'format_portfolio' in line:
        print(f"Line {i+1}: {line[:120]}")
