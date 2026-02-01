with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Update to handle both 'ticker' and 'symbol' keys
old_parse = '''                    # Parse JSON - extract tickers and weights and store in session state
                    ss.json_tickers = [item.get('ticker', '').upper() for item in data if item.get('ticker')]
                    ss.json_weights = {item.get('ticker', '').upper(): item.get('weight', 0) for item in data}
                    ss.json_names = {item.get('ticker', '').upper(): item.get('name', '') for item in data}'''

new_parse = '''                    # Parse JSON - extract tickers and weights and store in session state
                    # Support both 'ticker' and 'symbol' keys (IBKR format uses 'symbol')
                    ss.json_tickers = []
                    ss.json_weights = {}
                    ss.json_names = {}
                    
                    for item in data:
                        # Get ticker from either 'ticker' or 'symbol' field
                        ticker = item.get('ticker') or item.get('symbol') or ''
                        ticker = ticker.upper().strip()
                        
                        if ticker and ticker.isalpha():  # Skip numeric symbols like '2454'
                            ss.json_tickers.append(ticker)
                            ss.json_weights[ticker] = float(item.get('weight', 0) or 0)
                            # Get name from 'name' or 'originalName'
                            ss.json_names[ticker] = item.get('name') or item.get('originalName') or ''
                    
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_tickers = []
                    for t in ss.json_tickers:
                        if t not in seen:
                            seen.add(t)
                            unique_tickers.append(t)
                    ss.json_tickers = unique_tickers'''

if old_parse in content:
    content = content.replace(old_parse, new_parse)
    with open('../dashboard/portfolio_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Updated JSON parser to handle IBKR format (symbol key)")
else:
    print("❌ Could not find parse section")
    if 'ss.json_tickers' in content:
        print("Found ss.json_tickers in content")
