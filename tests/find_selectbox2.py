with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show context around selectbox lines
for i, line in enumerate(lines):
    if 'selectbox' in line.lower():
        print(f"\n--- Line {i+1} context ---")
        for j in range(max(0, i-5), min(len(lines), i+15)):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1}: {lines[j][:100]}")
