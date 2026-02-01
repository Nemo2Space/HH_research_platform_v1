with open('../dashboard/ai_pm/portfolio_intelligence.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix line 152
old1 = "signal = signal_data.get('signal', '').upper() if signal_data.get('signal') else ''"
new1 = "signal = str(signal_data.get('signal', '')).upper() if signal_data.get('signal') else ''"

# Fix line 233
old2 = "signal = (raw.get('signal') or raw.get('signal_type') or '').upper()"
new2 = "signal = str(raw.get('signal') or raw.get('signal_type') or '').upper()"

# Fix line 358
old3 = "signal = (raw.get('signal') or raw.get('signal_type') or '').upper()"
# Same as old2, will be replaced by the same operation

content = content.replace(old1, new1)
content = content.replace(old2, new2)

with open('../dashboard/ai_pm/portfolio_intelligence.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed float.upper() error in portfolio_intelligence.py")
