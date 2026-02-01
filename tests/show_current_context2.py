with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show context around lines 1050-1095
print("Lines 1040-1095:")
for i in range(1039, 1095):
    print(f"{i+1}: {lines[i]}")
