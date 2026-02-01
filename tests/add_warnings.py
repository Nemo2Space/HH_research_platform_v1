import re

with open('../dashboard/portfolio_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the ticker inclusion section and enhance it with detailed warnings
old_code = '''        # Handle ticker inclusion - if user specifies tickers, include ALL of them
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

new_code = '''        # Handle ticker inclusion - if user specifies tickers, include ALL of them
        if intent.tickers_include and len(intent.tickers_include) > 0:
            whitelist = [t.upper() for t in intent.tickers_include]
            
            # If restrict_to_tickers OR user provided 5+ specific tickers, use ONLY those
            if intent.restrict_to_tickers or len(whitelist) >= 5:
                # First, diagnose why tickers might be missing BEFORE filtering
                requested = set(whitelist)
                available_in_df = set(df['ticker'].str.upper())
                
                # Check for tickers not in database at all
                not_in_db = requested - available_in_df
                if not_in_db:
                    warnings.append(f"⚠️ Tickers not found in database: {', '.join(sorted(not_in_db))}")
                
                # Check for tickers with missing critical data
                tickers_in_df = df[df['ticker'].str.upper().isin(whitelist)]
                
                # Check market_cap issues
                if 'market_cap' in tickers_in_df.columns:
                    no_market_cap = tickers_in_df[tickers_in_df['market_cap'].isna()]['ticker'].tolist()
                    if no_market_cap:
                        warnings.append(f"⚠️ Tickers with no market cap data (may affect scoring): {', '.join(no_market_cap)}")
                
                # Check fundamental_score issues
                if 'fundamental_score' in tickers_in_df.columns:
                    no_fundamental = tickers_in_df[tickers_in_df['fundamental_score'].isna()]['ticker'].tolist()
                    if no_fundamental:
                        warnings.append(f"⚠️ Tickers with no fundamental scores: {', '.join(no_fundamental)}")
                
                # Now filter to whitelist
                df = df[df['ticker'].str.upper().isin(whitelist)]
                
                # Override holdings limits to include ALL specified tickers
                available_count = len(df)
                if available_count > 0:
                    constraints['min_holdings'] = available_count
                    constraints['max_holdings'] = available_count
                    
                # Summary warning if not all tickers included
                if available_count < len(whitelist):
                    warnings.append(f"📊 Requested {len(whitelist)} tickers, {available_count} available for portfolio construction")
            else:
                # Just ensure these tickers are included, but allow others too
                pass'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/portfolio_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Enhanced warnings added!")
else:
    print("❌ Could not find code block. Let me search...")
    for i, line in enumerate(content.split('\n')):
        if 'Handle ticker inclusion' in line:
            print(f"Found at line {i+1}")
