with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix to use price_map from session state for calculating market values
old_code = '''    # If no loaded statement, try to use snapshot positions from IBKR
    if not loaded_positions and snapshot:
        nav = getattr(snapshot, 'net_liquidation', 0) or 0
        positions = getattr(snapshot, 'positions', []) or []
        for p in positions:
            sym = (getattr(p, 'symbol', '') or '').strip().upper()
            if not sym:
                continue
            qty = getattr(p, 'quantity', 0) or getattr(p, 'position', 0) or 0
            mv = getattr(p, 'market_value', 0) or 0
            # Calculate market value from price if not available
            if mv == 0 and qty != 0:
                price = getattr(p, 'market_price', 0) or getattr(p, 'avgCost', 0) or 0
                if price > 0:
                    mv = abs(qty) * price
            weight = mv / nav if nav > 0 else 0
            loaded_positions[sym] = {
                'quantity': qty,
                'market_value': mv,
                'weight': weight,
            }'''

new_code = '''    # If no loaded statement, try to use snapshot positions from IBKR
    if not loaded_positions and snapshot:
        nav = getattr(snapshot, 'net_liquidation', 0) or 0
        positions = getattr(snapshot, 'positions', []) or []
        
        # Get price_map from session state if available
        price_map = st.session_state.get('ai_pm_price_map', {})
        
        for p in positions:
            sym = (getattr(p, 'symbol', '') or '').strip().upper()
            if not sym:
                continue
            qty = getattr(p, 'quantity', 0) or getattr(p, 'position', 0) or 0
            mv = getattr(p, 'market_value', 0) or 0
            
            # Calculate market value from price if not available
            if mv == 0 and qty != 0:
                # Try price_map first, then market_price, then avgCost
                price = price_map.get(sym, 0)
                if price == 0:
                    price = getattr(p, 'market_price', 0) or 0
                if price == 0:
                    price = getattr(p, 'avgCost', 0) or 0
                if price > 0:
                    mv = abs(qty) * price
            
            weight = mv / nav if nav > 0 else 0
            loaded_positions[sym] = {
                'quantity': qty,
                'market_value': mv,
                'weight': weight,
            }'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed comparison to use price_map")
else:
    print("❌ Could not find the section")
