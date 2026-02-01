with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find where portfolio_builder_last_result is displayed
print("Searching for portfolio result display...")
for i, line in enumerate(lines):
    if 'portfolio_builder_last_result' in line and ('display' in line.lower() or 'render' in line.lower() or 'show' in line.lower() or 'markdown' in line.lower()):
        print(f"\nLine {i+1}: {line[:120]}")

print("\n\nSearching for enhanced_portfolio_display...")
for i, line in enumerate(lines):
    if 'enhanced_portfolio' in line.lower():
        print(f"Line {i+1}: {line[:120]}")
