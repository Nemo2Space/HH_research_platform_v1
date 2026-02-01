with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the JSON import build section and check current state
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'Build Portfolio from JSON' in line:
        print(f"Found at line {i+1}")
        # Show context
        for j in range(max(0, i-5), min(len(lines), i+50)):
            print(f"{j+1}: {lines[j][:120]}")
        break
