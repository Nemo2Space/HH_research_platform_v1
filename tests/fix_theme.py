with open('../dashboard/portfolio_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the theme filter condition
old_code = '''        # Theme-based filtering (AI, Semiconductors, etc.)
        if intent.theme and intent.require_theme_match:
            df, theme_warnings = self._apply_theme_filter(df, intent, user_request)
            warnings.extend(theme_warnings)'''

new_code = '''        # Theme-based filtering (AI, Semiconductors, etc.)
        # SKIP theme filter if user explicitly specified tickers with restrict_to_tickers
        if intent.theme and intent.require_theme_match:
            if intent.restrict_to_tickers and intent.tickers_include and len(intent.tickers_include) >= 5:
                # User specified exact tickers - skip theme filter, respect their choices
                logger.info(f"Skipping theme filter - user specified {len(intent.tickers_include)} tickers with restrict_to_tickers=True")
            else:
                df, theme_warnings = self._apply_theme_filter(df, intent, user_request)
                warnings.extend(theme_warnings)'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/portfolio_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed! Theme filter now skipped when user specifies tickers.")
else:
    print("❌ Could not find exact code. Searching...")
    for i, line in enumerate(content.split('\n')):
        if 'Theme-based filtering' in line:
            print(f"Line {i+1}: {line}")
