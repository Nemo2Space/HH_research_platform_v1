with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find "if confirm:" and get much more context
idx = content.find('if confirm:')
if idx > 0:
    # Get 8000 chars to see the full flow
    end = min(len(content), idx + 8000)
    print(content[idx:end])
