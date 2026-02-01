with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Show lines 3920-4000
print("Execution code (lines 3920-4000):")
print("="*70)
for i in range(3919, min(4000, len(lines))):
    print(f"{i+1}: {lines[i]}")
