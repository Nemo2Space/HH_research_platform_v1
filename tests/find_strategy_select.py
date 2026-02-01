with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find where Strategy dropdown is and add Saved Portfolio selector nearby
# Look for the strategy selectbox area

lines = content.split('\n')
for i, line in enumerate(lines):
    if 'Strategy' in line and 'selectbox' in line.lower():
        print(f"Line {i+1}: {line[:100]}")
        for j in range(max(0, i-10), min(len(lines), i+20)):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1}: {lines[j][:100]}")
        break
