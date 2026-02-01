with open('../dashboard/ai_pm/ibkr_gateway.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find get_account_summary and show full function
in_func = False
for i, line in enumerate(lines):
    if 'def get_account_summary' in line:
        in_func = True
    if in_func:
        print(f"{i+1}: {line}")
        if line.strip().startswith('def ') and 'get_account_summary' not in line:
            break
