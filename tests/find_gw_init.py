with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find gateway init patterns
patterns = [
    'gw = IbkrGateway()',
    'st.session_state.ai_pm_gateway = gw',
    'ai_pm_connection_ok',
]

lines = content.split('\n')
for i, line in enumerate(lines):
    for p in patterns:
        if p in line:
            print(f"Line {i+1}: {line.strip()[:80]}")
