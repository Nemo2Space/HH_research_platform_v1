with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find the Run Now pipeline section starting at line 3442
print("Run Now Pipeline Code (lines 3442-3550):")
print("="*70)
for i in range(3441, min(3550, len(lines))):
    print(f"{i+1}: {lines[i]}")
