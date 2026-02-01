with open('../dashboard/portfolio_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find PortfolioHolding class definition
print("Searching for PortfolioHolding class...")
for i, line in enumerate(lines):
    if 'class PortfolioHolding' in line or '@dataclass' in line:
        print(f"\nLine {i+1}: {line}")
        # Show the class definition
        for j in range(i, min(len(lines), i+40)):
            print(f"{j+1}: {lines[j]}")
            if j > i and lines[j].strip() and not lines[j].startswith(' ') and not lines[j].startswith('\t'):
                break
