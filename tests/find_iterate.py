with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the section
idx = content.find('iterating')
if idx > 0:
    print("Found 'iterating' at:", idx)
    print(repr(content[idx:idx+600]))
