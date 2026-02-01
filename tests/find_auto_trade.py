with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

for i, line in enumerate(lines):
    if 'Auto Trade' in line:
        print(f"\n--- Line {i+1} context ---")
        for j in range(max(0, i-10), min(len(lines), i+20)):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1}: {lines[j]}")
        break
