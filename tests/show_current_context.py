with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show context around lines 1100-1140
print("Lines 1090-1140:")
for i in range(1089, 1140):
    print(f"{i+1}: {lines[i]}")
