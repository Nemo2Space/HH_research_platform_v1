with open('../dashboard/ai_pm/rebalance_metrics.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the position_map building to use price_map for market_value calculation
old_code = '''    # Build position map
    positions = getattr(snapshot, 'positions', []) or []
    position_map = {}
    for p in positions:
        sym = (getattr(p, 'symbol', '') or '').strip().upper()
        if sym:
            position_map[sym] = {
                'quantity': _safe_float(getattr(p, 'quantity', 0)),
                'market_value': _safe_float(getattr(p, 'market_value', 0)),
            }'''

new_code = '''    # Build position map
    positions = getattr(snapshot, 'positions', []) or []
    position_map = {}
    for p in positions:
        sym = (getattr(p, 'symbol', '') or '').strip().upper()
        if sym:
            qty = _safe_float(getattr(p, 'quantity', 0)) or _safe_float(getattr(p, 'position', 0))
            # Try to get market_value directly, or calculate from price
            mv = _safe_float(getattr(p, 'market_value', 0))
            if mv == 0 and qty != 0:
                # Calculate from price_map if available
                price = price_map.get(sym, 0)
                if price > 0:
                    mv = abs(qty) * price
                else:
                    # Fall back to avgCost * quantity
                    avg_cost = _safe_float(getattr(p, 'avgCost', 0)) or _safe_float(getattr(p, 'average_cost', 0))
                    if avg_cost > 0:
                        mv = abs(qty) * avg_cost
            position_map[sym] = {
                'quantity': qty,
                'market_value': mv,
            }'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/rebalance_metrics.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed position_map to calculate market_value from price")
else:
    print("❌ Could not find the section to replace")
