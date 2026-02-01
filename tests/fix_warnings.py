with open('../dashboard/portfolio_engine.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with "Handle ticker inclusion"
start_line = None
for i, line in enumerate(lines):
    if 'Handle ticker inclusion' in line:
        start_line = i
        break

if start_line is None:
    print("❌ Could not find 'Handle ticker inclusion'")
    exit()

print(f"Found at line {start_line + 1}")

# Show current code (lines start_line to start_line + 30)
print("\nCurrent code:")
print("="*60)
for i in range(start_line, min(start_line + 35, len(lines))):
    print(f"{i+1}: {lines[i]}", end='')
print("\n" + "="*60)

# Now let's replace from line 1640 onwards
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
                tickers_in_df = df[df['ticker'].str.upper().isin(whitelist)].copy()
                
                # Check market_cap issues
                if 'market_cap' in tickers_in_df.columns:
                    no_market_cap = tickers_in_df[tickers_in_df['market_cap'].isna()]['ticker'].tolist()
                    if no_market_cap:
                        warnings.append(f"⚠️ Tickers with no market cap data: {', '.join(no_market_cap)}")
                
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
                    warnings.append(f"📊 Requested {len(whitelist)} tickers, only {available_count} available for portfolio")
            else:
                # Just ensure these tickers are included, but allow others too
                pass

'''

# Find the end of the current ticker inclusion block
# Look for the next section (sectors_include or tickers_exclude)
end_line = start_line + 1
for i in range(start_line + 1, len(lines)):
    if 'if intent.tickers_exclude' in lines[i] or 'if intent.sectors_include' in lines[i]:
        end_line = i
        break

print(f"\nReplacing lines {start_line + 1} to {end_line}")

# Replace the code
new_lines = lines[:start_line] + [new_code] + lines[end_line:]

with open('../dashboard/portfolio_engine.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ Enhanced warnings applied!")
