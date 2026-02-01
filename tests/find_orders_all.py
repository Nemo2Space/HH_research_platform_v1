with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the exact text
idx = content.find('orders = orders_all')
if idx > 0:
    print("Code around 'orders = orders_all':")
    print(repr(content[idx:idx+300]))
