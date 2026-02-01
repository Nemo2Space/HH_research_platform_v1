with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show lines 4000-4100
print("Execution code (lines 4000-4100):")
print("="*70)
for i in range(3999, min(4100, len(lines))):
    print(f"{i+1}: {lines[i]}")
