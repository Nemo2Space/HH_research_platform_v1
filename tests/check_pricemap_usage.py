with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the function signature and see how price_map is used
idx = content.find('def execute_trade_plan(')
if idx > 0:
    print("Function signature and price_map usage:")
    print("="*70)
    # Get the function
    print(content[idx:idx+2000])
