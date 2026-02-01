with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the JSON import to set the correct session state key
old_code = '''                                            # Store in session state
                                            ss.portfolio_builder_last_result = result
                                            ss.portfolio_builder_loaded_info = None
                                            ss.portfolio_builder_loaded_holdings = None
                                            ss.portfolio_builder_last_errors = []'''

new_code = '''                                            # Store in session state - BOTH keys needed!
                                            ss.portfolio_builder_last_result = result
                                            ss.last_portfolio_result = result  # This triggers enhanced display
                                            ss.portfolio_builder_loaded_info = None
                                            ss.portfolio_builder_loaded_holdings = None
                                            ss.portfolio_builder_last_errors = []'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/portfolio_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed JSON import to trigger enhanced display!")
else:
    print("❌ Could not find the code block")
    # Let's search for what we have
    if 'ss.portfolio_builder_last_result = result' in content:
        print("Found portfolio_builder_last_result assignment")
    if 'ss.last_portfolio_result' in content:
        print("Found last_portfolio_result already in code")
