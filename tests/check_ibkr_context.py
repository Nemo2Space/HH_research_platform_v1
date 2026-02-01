with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check context around these IBKR calls
positions = [72202, 73412, 72548]
for pos in positions:
    # Find the function name this is in
    func_start = content.rfind('def ', 0, pos)
    func_name_end = content.find('(', func_start)
    func_name = content[func_start:func_name_end]
    print(f"\nAt {pos}: {func_name}")
    print(f"Context: {content[pos-50:pos+100]}")
