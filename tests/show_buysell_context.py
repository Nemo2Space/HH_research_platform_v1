with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show context around line 1350
print("Lines 1345-1360:")
for i in range(1344, 1360):
    print(f"{i+1}: {lines[i]}")
