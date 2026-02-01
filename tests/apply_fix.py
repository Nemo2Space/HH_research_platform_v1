import re

# Read the file
with open('../dashboard/portfolio_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the restrict_to_tickers section
old_code = '''        if intent.restrict_to_tickers and intent.tickers_include:
            whitelist = [t.upper() for t in intent.tickers_include]
            df = df[df['ticker'].str.upper().isin(whitelist)]'''

new_code = '''        if intent.restrict_to_tickers and intent.tickers_include:
            whitelist = [t.upper() for t in intent.tickers_include]
            df = df[df['ticker'].str.upper().isin(whitelist)]
            
            # CRITICAL: Override holdings limits to include ALL specified tickers
            available_count = len(df)
            if available_count > 0:
                constraints['min_holdings'] = available_count
                constraints['max_holdings'] = available_count
            
            # Warn about missing tickers
            requested = set(whitelist)
            available = set(df['ticker'].str.upper())
            missing_tickers = requested - available
            if missing_tickers:
                warnings.append(f"Tickers not in database: {', '.join(sorted(missing_tickers))}")'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/portfolio_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fix applied successfully!")
else:
    print("❌ Could not find the code to replace. May already be fixed or different format.")
    print("\nSearching for 'restrict_to_tickers'...")
    for i, line in enumerate(content.split('\n')):
        if 'restrict_to_tickers' in line:
            print(f"  Line {i+1}: {line[:80]}")
