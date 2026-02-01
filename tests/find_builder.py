with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find where AI Portfolio Builder tab is rendered
print("Searching for AI Portfolio Builder section...")
for i, line in enumerate(lines):
    if 'AI Portfolio Builder' in line or 'portfolio_builder' in line.lower():
        print(f"\nLine {i+1}: {line[:100]}")
        # Show some context
        for j in range(max(0, i-2), min(len(lines), i+5)):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1}: {lines[j][:100]}")
        print()
