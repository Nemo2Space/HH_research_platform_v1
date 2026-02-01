with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check context around _ticker_price call at 73078
pos = 73078
func_start = content.rfind('def ', 0, pos)
func_name_end = content.find('(', func_start)
func_name = content[func_start:func_name_end]
print(f"_ticker_price call is in: {func_name}")
print(f"\nBroader context:")
print(content[pos-300:pos+200])
