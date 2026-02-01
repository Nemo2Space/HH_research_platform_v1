with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and disable IBKR fallback in Run Now section (occurrence 3)
old_code = '''                # FALLBACK: IBKR for remaining missing (only if few missing)
                still_missing = [s for s in universe if s not in price_map or not price_map.get(s)]
                
                if still_missing and len(still_missing) <= 20:  # Only use IBKR for small sets
                    try:
                        ibkr_prices, _ = _fetch_ibkr_price_map(
                            ib=gw.ib,
                            symbols=still_missing,
                            currency=(snapshot.currency or "USD"),
                            exchange="SMART",
                        )
                        for sym, price in ibkr_prices.items():
                            if price and price > 0:
                                price_map[sym] = price
                                price_diag['source_counts']['ibkr'] += 1
                    except Exception as e:
                        pass  # Silent fail - Yahoo should have most prices
                elif still_missing:
                    price_diag['missing'] = still_missing[:20]'''

new_code = '''                # IBKR fallback DISABLED - causes freeze in Streamlit threads
                still_missing = [s for s in universe if s not in price_map or not price_map.get(s)]
                if still_missing:
                    price_diag['missing'] = still_missing[:20]'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Removed IBKR fallback from Run Now section")
else:
    print("❌ Could not find exact code, searching for pattern...")
    # Try to find it
    idx = content.find('FALLBACK: IBKR for remaining missing')
    if idx > 0:
        print(f"Found at {idx}:")
        print(repr(content[idx:idx+600]))
