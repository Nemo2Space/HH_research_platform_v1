with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find _make_contract to see how contracts are built
idx = content.find('def _make_contract')
if idx > 0:
    print("_make_contract function:")
    print(content[idx:idx+800])
