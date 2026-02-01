with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the quote fetch section
idx = content.find('quote = _fetch_quote')
if idx > 0:
    print("Found quote fetch at index", idx)
    print(repr(content[idx:idx+400]))
else:
    print("Not found")
