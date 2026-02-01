with open('../dashboard/portfolio_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

print("Finding theme filter application...")
for i, line in enumerate(lines):
    if 'theme' in line.lower() and 'filter' in line.lower():
        print(f"\nLine {i+1}: {line}")
        # Show context
        for j in range(max(0, i-3), min(len(lines), i+5)):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1}: {lines[j][:100]}")
