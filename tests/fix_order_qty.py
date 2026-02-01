with open('../dashboard/ai_pm/trade_planner.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the order quantity calculation to round to whole shares and validate
old_code = '''        qty = delta_value / float(px)
        qty = _clamp_qty(qty)
        if qty == 0.0:
            continue

        notional = abs(qty) * float(px)
        if notional < float(constraints.min_order_notional_usd):
            continue

        action = "BUY" if qty > 0 else "SELL"
        qty_abs = abs(qty)

        reason = f"Rebalance: current={cw:.3f}, target={tw:.3f}, drift={d:.3f}"
        orders.append(
            OrderTicket(
                symbol=sym,
                action=action,
                quantity=qty_abs,
                order_type="MKT",
                limit_price=None,
                tif="DAY",
                reason=reason,
            )
        )'''

new_code = '''        qty = delta_value / float(px)
        qty = _clamp_qty(qty)
        if qty == 0.0:
            continue

        # Round to whole shares - use floor for sells (conservative), ceil for buys
        if qty > 0:
            qty_rounded = math.ceil(qty)  # BUY: round up
        else:
            qty_rounded = math.floor(qty)  # SELL: round down (more negative)
        
        # Skip if rounded qty is 0
        if qty_rounded == 0:
            continue

        action = "BUY" if qty_rounded > 0 else "SELL"
        qty_abs = abs(qty_rounded)
        
        # VALIDATION: For SELL orders, ensure qty doesn't exceed current position
        if action == "SELL":
            current_pos = pos_map.get(sym)
            current_qty = int(getattr(current_pos, 'quantity', 0) or 0) if current_pos else 0
            if qty_abs > current_qty:
                if current_qty > 0:
                    qty_abs = current_qty  # Cap at current position
                    warnings.append(f"{sym}: Sell qty capped to position size ({current_qty})")
                else:
                    warnings.append(f"{sym}: Cannot sell - no position found")
                    continue

        notional = qty_abs * float(px)
        if notional < float(constraints.min_order_notional_usd):
            continue

        reason = f"Rebalance: current={cw:.3f}, target={tw:.3f}, drift={d:.3f}"
        orders.append(
            OrderTicket(
                symbol=sym,
                action=action,
                quantity=qty_abs,  # Now always a whole number
                order_type="MKT",
                limit_price=None,
                tif="DAY",
                reason=reason,
            )
        )'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/trade_planner.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed order quantities - now rounded to whole shares with validation!")
else:
    print("❌ Could not find the section to replace")
