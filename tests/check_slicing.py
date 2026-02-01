with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find max_slice logic
idx = content.find('max_slice')
if idx > 0:
    print("Order slicing logic:")
    print(content[idx:idx+1000])
