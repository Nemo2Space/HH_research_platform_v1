with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find the "No portfolio loaded" message
for i, line in enumerate(lines):
    if 'No portfolio loaded for' in line:
        print(f"\nLine {i+1}: {line[:100]}")
        for j in range(max(0, i-10), min(len(lines), i+5)):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1}: {lines[j][:100]}")
