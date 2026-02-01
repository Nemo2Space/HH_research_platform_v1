with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find what comes after cash line
idx = content.find('cash = _safe_float(snapshot.total_cash)')
if idx > 0:
    print("Code after cash line:")
    print(content[idx:idx+1500])
