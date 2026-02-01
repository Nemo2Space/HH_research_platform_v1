with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the comparison to use snapshot positions when loaded_positions is empty
old_code = '''def _render_portfolio_comparison(snapshot, targets, signals, account: str) -> None:
    """
    Render a comparison table showing:
    - Current holdings from loaded portfolio statement
    - Target weights from AI PM
    - Actions needed (buy more, sell, hold, new position)
    """
    st.markdown("### 📊 Portfolio Comparison: Current vs Target")

    # Get current positions from loaded statement
    loaded_positions = _get_loaded_portfolio_positions(account)

    # Get target weights
    target_weights = targets.weights if targets and targets.weights else {}

    if not loaded_positions and not target_weights:
        st.info("No positions or targets available for comparison.")
        return

    # Build comparison data
    all_symbols = set(loaded_positions.keys()) | set(target_weights.keys())'''

new_code = '''def _render_portfolio_comparison(snapshot, targets, signals, account: str) -> None:
    """
    Render a comparison table showing:
    - Current holdings from loaded portfolio statement OR IBKR live positions
    - Target weights from AI PM
    - Actions needed (buy more, sell, hold, new position)
    """
    st.markdown("### 📊 Portfolio Comparison: Current vs Target")

    # Get current positions from loaded statement
    loaded_positions = _get_loaded_portfolio_positions(account)
    
    # If no loaded statement, try to use snapshot positions from IBKR
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
            }

    # Get target weights
    target_weights = targets.weights if targets and targets.weights else {}

    if not loaded_positions and not target_weights:
        st.info("No positions or targets available for comparison.")
        return

    # Build comparison data
    all_symbols = set(loaded_positions.keys()) | set(target_weights.keys())'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed Portfolio Comparison to use IBKR positions")
else:
    print("❌ Could not find the section to replace")
