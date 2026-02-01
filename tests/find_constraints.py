with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find DEFAULT_CONSTRAINTS usage
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'DEFAULT_CONSTRAINTS' in line or 'RiskConstraints' in line:
        print(f"{i+1}: {line.strip()[:90]}")
