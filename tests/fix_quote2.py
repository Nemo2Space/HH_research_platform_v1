with open('../dashboard/ai_pm/execution_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_quote_fetch = '''quote = _fetch_quote(ib, contract, timeout_sec=2.0)

        # If quote is totally missing, SKIP (your policy: missing prices -> skip)
        if quote.get("bid") is None and quote.get("ask") is None and quote.get("last") is None and quote.get(
                "mid") is None:
            notes.append(f"{sym}: missing live quote (bid/ask/last unavailable); skipped execution.")
            continue'''

new_quote_fetch = '''# Use pre-fetched price if available (much faster than IBKR reqMktData)
        if skip_live_quotes and price_map and sym in price_map and price_map[sym]:
            pre_price = float(price_map[sym])
            quote = {"bid": pre_price * 0.999, "ask": pre_price * 1.001, "last": pre_price, "mid": pre_price}
        else:
            quote = _fetch_quote(ib, contract, timeout_sec=2.0)

        # If quote is totally missing, SKIP (your policy: missing prices -> skip)
        if quote.get("bid") is None and quote.get("ask") is None and quote.get("last") is None and quote.get(
                "mid") is None:
            notes.append(f"{sym}: missing live quote (bid/ask/last unavailable); skipped execution.")
            continue'''

if old_quote_fetch in content:
    content = content.replace(old_quote_fetch, new_quote_fetch)
    with open('../dashboard/ai_pm/execution_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed quote fetching to use pre-fetched prices")
else:
    print("❌ Could not find - showing what's there:")
    idx = content.find('quote = _fetch_quote')
    print(repr(content[idx:idx+500]))
