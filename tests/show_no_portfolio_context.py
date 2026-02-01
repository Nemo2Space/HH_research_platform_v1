with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show more context around the no portfolio message
print("Lines 2455-2485:")
for i in range(2454, 2485):
    print(f"{i+1}: {lines[i]}")
