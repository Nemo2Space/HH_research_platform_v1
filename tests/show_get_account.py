with open('../dashboard/ai_pm/ibkr_gateway.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find get_account_summary and show it
in_func = False
start_line = 0
for i, line in enumerate(lines):
    if 'def get_account_summary' in line:
        in_func = True
        start_line = i
    if in_func:
        print(f"{i+1}: {line}")
        # Stop at next function or after 80 lines
        if i > start_line + 80:
            break
        if line.strip().startswith('def ') and i > start_line:
            break
