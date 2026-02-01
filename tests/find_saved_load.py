with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find where saved portfolio is loaded
print("Searching for saved portfolio loading...")
for i, line in enumerate(lines):
    if 'portfolio_builder_loaded_info' in line and '=' in line and 'info' in line:
        print(f"\nLine {i+1}: {line[:120]}")
        for j in range(max(0, i-5), min(len(lines), i+10)):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1}: {lines[j][:120]}")
