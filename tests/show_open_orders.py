with open('../dashboard/ai_pm/ibkr_gateway.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find get_open_orders
for i, line in enumerate(lines):
    if 'def get_open_orders' in line:
        print(f"Found at line {i+1}:")
        for j in range(i, min(i+40, len(lines))):
            print(f"{j+1}: {lines[j]}")
        break
