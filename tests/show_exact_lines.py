with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show exact lines around line 2707
print("Lines 2700-2720:")
for i in range(2699, 2720):
    print(f"{i+1}: {lines[i]}")
