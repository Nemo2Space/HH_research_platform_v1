with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The issue is that json_tickers, json_weights, found_tickers are local variables
# that get lost on rerun. We need to store them in session state too.

# Find where json_tickers is defined and store it
old_parse = '''                if isinstance(data, list) and len(data) > 0:
                    # Parse JSON - extract tickers and weights
                    json_tickers = [item.get('ticker', '').upper() for item in data if item.get('ticker')]
                    json_weights = {item.get('ticker', '').upper(): item.get('weight', 0) for item in data}
                    json_names = {item.get('ticker', '').upper(): item.get('name', '') for item in data}
                    
                    st.success(f"✅ Found {len(json_tickers)} tickers in JSON")'''

new_parse = '''                if isinstance(data, list) and len(data) > 0:
                    # Parse JSON - extract tickers and weights and store in session state
                    ss.json_tickers = [item.get('ticker', '').upper() for item in data if item.get('ticker')]
                    ss.json_weights = {item.get('ticker', '').upper(): item.get('weight', 0) for item in data}
                    ss.json_names = {item.get('ticker', '').upper(): item.get('name', '') for item in data}
                    
                    st.success(f"✅ Found {len(ss.json_tickers)} tickers in JSON")'''

if old_parse in content:
    content = content.replace(old_parse, new_parse)
    print("✅ Fixed parse section")
else:
    print("❌ Could not find parse section")

# Now replace all references to json_tickers, json_weights, json_names with ss. versions
content = content.replace('json_tickers', 'ss.json_tickers')
content = content.replace('ss.ss.json_tickers', 'ss.json_tickers')  # Fix double prefix
content = content.replace('json_weights', 'ss.json_weights')
content = content.replace('ss.ss.json_weights', 'ss.json_weights')
content = content.replace('json_names', 'ss.json_names')
content = content.replace('ss.ss.json_names', 'ss.json_names')

# Also store found_tickers and missing_tickers
old_check = '''                    # Check which tickers are in the database
                    if ss.portfolio_builder_df is not None:
                        available = set(ss.portfolio_builder_df['ticker'].str.upper())
                        found_tickers = [t for t in ss.json_tickers if t in available]
                        missing_tickers = [t for t in ss.json_tickers if t not in available]'''

new_check = '''                    # Check which tickers are in the database
                    if ss.portfolio_builder_df is not None:
                        available = set(ss.portfolio_builder_df['ticker'].str.upper())
                        ss.json_found_tickers = [t for t in ss.json_tickers if t in available]
                        ss.json_missing_tickers = [t for t in ss.json_tickers if t not in available]'''

if old_check in content:
    content = content.replace(old_check, new_check)
    print("✅ Fixed check section")
else:
    print("❌ Could not find check section")

# Replace found_tickers and missing_tickers with session state versions
content = content.replace('found_tickers', 'ss.json_found_tickers')
content = content.replace('ss.json_ss.json_found_tickers', 'ss.json_found_tickers')
content = content.replace('missing_tickers', 'ss.json_missing_tickers')
content = content.replace('ss.json_ss.json_missing_tickers', 'ss.json_missing_tickers')

with open('../dashboard/portfolio_tab.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ All JSON variables now use session state!")
