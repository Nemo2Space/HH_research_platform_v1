with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Search for any connect/disconnect calls
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'connect' in line.lower() and ('gw.' in line or 'gateway' in line.lower() or 'ib.' in line):
        print(f"{i+1}: {line.strip()[:90]}")
