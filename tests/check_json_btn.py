with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the JSON build button section
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'pb_build_from_json' in line:
        print(f"Found at line {i+1}")
        for j in range(max(0, i-5), min(len(lines), i+40)):
            print(f"{j+1}: {lines[j][:120]}")
        break
