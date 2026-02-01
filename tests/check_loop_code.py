with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

idx = content.find('looping')
if idx > 0:
    print("Code around 'looping':")
    print(content[idx-50:idx+500])
