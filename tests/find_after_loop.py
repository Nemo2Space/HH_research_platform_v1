with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find what comes after the cash projection loop
idx = content.find('cash projection order')
if idx > 0:
    # Find end of that if block and what follows
    print("Code AFTER cash projection loop:")
    print(content[idx:idx+1500])
