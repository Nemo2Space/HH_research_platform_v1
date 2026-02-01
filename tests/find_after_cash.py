with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find what comes after the cash policy section
idx = content.find('checking cash policy')
if idx > 0:
    print("Code after 'checking cash policy':")
    print(content[idx:idx+2000])
