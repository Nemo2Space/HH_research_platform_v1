with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find yfinance download in the Run Now section
idx = content.find('yf.download')
if idx > 0:
    print(f"Found yf.download at {idx}")
    print(content[idx-200:idx+800])
