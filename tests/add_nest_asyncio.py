with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add nest_asyncio at the very start of execute_trade_plan
old_start = '''def execute_trade_plan(
        *,
        ib: IB,'''

new_start = '''def execute_trade_plan(
        *,
        ib: IB,'''

# Find where the function body starts and add nest_asyncio there
idx = content.find('"""Execute orders for a plan.')
if idx > 0:
    # Find the end of the docstring
    end_doc = content.find('"""', idx + 10)
    if end_doc > 0:
        # Insert nest_asyncio after docstring
        old_after_doc = content[end_doc:end_doc+100]
        print(f"After docstring: {repr(old_after_doc)}")

# Let's just add it at the top of the function after imports
old_code = '''    import logging
    import sys
    _log = logging.getLogger(__name__)'''

new_code = '''    import logging
    import sys
    import nest_asyncio
    nest_asyncio.apply()
    _log = logging.getLogger(__name__)'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added nest_asyncio.apply() at start of execute_trade_plan")
else:
    print("❌ Could not find code")
