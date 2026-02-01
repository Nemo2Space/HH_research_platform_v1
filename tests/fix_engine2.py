import re

with open('../dashboard/portfolio_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the section we need to update
old_code = '''        if intent.restrict_to_tickers and intent.tickers_include:
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

new_code = '''        # Handle ticker inclusion - if user specifies tickers, include ALL of them
        if intent.tickers_include and len(intent.tickers_include) > 0:
            whitelist = [t.upper() for t in intent.tickers_include]
            
            # If restrict_to_tickers OR user provided 5+ specific tickers, use ONLY those
            if intent.restrict_to_tickers or len(whitelist) >= 5:
                df = df[df['ticker'].str.upper().isin(whitelist)]
                
                # Override holdings limits to include ALL specified tickers
                available_count = len(df)
                if available_count > 0:
                    constraints['min_holdings'] = available_count
                    constraints['max_holdings'] = available_count
                
                # Warn about missing tickers
                requested = set(whitelist)
                available = set(df['ticker'].str.upper())
                missing_tickers = requested - available
                if missing_tickers:
                    warnings.append(f"Tickers not in database: {', '.join(sorted(missing_tickers))}")
            else:
                # Just ensure these tickers are included, but allow others too
                pass'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/portfolio_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Enhanced fix applied! Now includes all tickers when 5+ are specified.")
else:
    print("❌ Previous fix not found. Searching for alternatives...")
    
    # Try to find the original code
    original = '''        if intent.restrict_to_tickers and intent.tickers_include:
            whitelist = [t.upper() for t in intent.tickers_include]
            df = df[df['ticker'].str.upper().isin(whitelist)]'''
    
    if original in content:
        content = content.replace(original, new_code)
        with open('../dashboard/portfolio_engine.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Applied fix to original code!")
    else:
        print("Searching for 'tickers_include'...")
        for i, line in enumerate(content.split('\n')):
            if 'tickers_include' in line and 'intent' in line:
                print(f"  Line {i+1}: {line[:80]}")
