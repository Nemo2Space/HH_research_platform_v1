with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the execute function
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'def execute' in line or 'def send_order' in line or 'placeOrder' in line or 'reqAllOpenOrders' in line:
        print(f"{i+1}: {line.strip()[:90]}")
