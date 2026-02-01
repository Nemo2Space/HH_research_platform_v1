with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find where build_target_weights is called and where we can intercept with saved portfolio
for i, line in enumerate(lines):
    if 'build_target_weights' in line:
        print(f"\n--- Line {i+1} context ---")
        for j in range(max(0, i-10), min(len(lines), i+25)):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1}: {lines[j][:110]}")
