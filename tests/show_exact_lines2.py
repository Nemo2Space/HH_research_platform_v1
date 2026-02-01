with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show exact lines around line 2854
print("Lines 2850-2875:")
for i in range(2849, 2875):
    print(f"{i+1}: {lines[i]}")
