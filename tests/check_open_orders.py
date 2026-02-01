with open('../dashboard/ai_pm/ibkr_gateway.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the exact text around get_open_orders
idx = content.find('def get_open_orders')
if idx > 0:
    print("Found get_open_orders, showing surrounding text:")
    print(repr(content[idx:idx+600]))
else:
    print("Function not found")
