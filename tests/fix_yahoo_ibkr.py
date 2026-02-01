with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the IBKR fallback from _fetch_price_map_yahoo_first
old_code = '''    # FALLBACK: IBKR for missing (only if few and ib available)
    missing = [s for s in syms if s not in price_map]
    if missing and ib and len(missing) <= 20:
        try:
            from ib_insync import Stock
            contracts = [Stock(sym, 'SMART', 'USD') for sym in missing[:20]]
            qualified = ib.qualifyContracts(*contracts)
            tickers = ib.reqTickers(*qualified)
            for t in (tickers or []):
                try:
                    sym = (t.contract.symbol or "").upper()
                    for attr in ("marketPrice", "last", "close"):
                        val = getattr(t, attr, None)
                        if callable(val): val = val()
                        if val and val > 0:
                            price_map[sym] = float(val)
                            diag["source_counts"]["ibkr"] += 1
                            break
                except:
                    pass
        except:
            pass'''

new_code = '''    # IBKR fallback DISABLED - causes freeze in Streamlit threads
    # Yahoo Finance is fast and sufficient for most symbols
    missing = [s for s in syms if s not in price_map]
    if missing:
        diag["source_counts"]["ibkr"] = 0  # Skipped'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Removed IBKR fallback from _fetch_price_map_yahoo_first")
else:
    print("❌ Could not find IBKR fallback code")
