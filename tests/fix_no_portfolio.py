with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Update to show IBKR live positions when no statement is loaded
old_code = '''        loaded_positions = _get_loaded_portfolio_positions(sel_account)
        if loaded_positions:
            with st.expander(f"📂 Loaded Portfolio: {sel_account} ({len(loaded_positions)} positions)", expanded=False):
                total_value = sum(p['market_value'] for p in loaded_positions.values())
                st.metric("Total Loaded Portfolio Value", f"")

                # Top 10 positions
                sorted_pos = sorted(loaded_positions.items(), key=lambda x: x[1]['market_value'], reverse=True)[:10]
                pos_df = pd.DataFrame([
                    {
                        'Symbol': sym,
                        'Weight %': f"{data['weight'] * 100:.2f}%",
                        'Market Value': f"",
                        'P&L': f"",
                        'P&L %': f"{data['unrealized_pnl_pct']:.1f}%"
                    }
                    for sym, data in sorted_pos
                ])
                st.dataframe(pos_df, width='stretch', hide_index=True)

                st.caption(
                    "💡 This is your current portfolio from the uploaded statement. The AI PM will consider these positions when building targets.")
        else:
            st.info(
                f"📂 No portfolio loaded for {sel_account}. Upload a statement in the Portfolio tab to see current holdings.")'''

new_code = '''        loaded_positions = _get_loaded_portfolio_positions(sel_account)
        if loaded_positions:
            with st.expander(f"📂 Loaded Portfolio: {sel_account} ({len(loaded_positions)} positions)", expanded=False):
                total_value = sum(p['market_value'] for p in loaded_positions.values())
                st.metric("Total Loaded Portfolio Value", f"")

                # Top 10 positions
                sorted_pos = sorted(loaded_positions.items(), key=lambda x: x[1]['market_value'], reverse=True)[:10]
                pos_df = pd.DataFrame([
                    {
                        'Symbol': sym,
                        'Weight %': f"{data['weight'] * 100:.2f}%",
                        'Market Value': f"",
                        'P&L': f"",
                        'P&L %': f"{data['unrealized_pnl_pct']:.1f}%"
                    }
                    for sym, data in sorted_pos
                ])
                st.dataframe(pos_df, width='stretch', hide_index=True)

                st.caption(
                    "💡 This is your current portfolio from the uploaded statement. The AI PM will consider these positions when building targets.")
        else:
            # Check for IBKR live positions
            ibkr_positions = []
            if gw.is_connected():
                try:
                    ibkr_positions = gw.ib.positions(account=sel_account) if hasattr(gw.ib, 'positions') else []
                except Exception:
                    ibkr_positions = []
            
            if ibkr_positions:
                with st.expander(f"📡 Live IBKR Positions: {sel_account} ({len(ibkr_positions)} positions)", expanded=False):
                    ibkr_df = pd.DataFrame([
                        {
                            'Symbol': p.contract.symbol,
                            'Position': int(p.position),
                            'Avg Cost': f"" if p.avgCost else "N/A",
                        }
                        for p in ibkr_positions[:20]
                    ])
                    st.dataframe(ibkr_df, width='stretch', hide_index=True)
                    st.caption("💡 Live positions from IBKR. Click Run Now to build rebalancing plan.")
            else:
                st.info(
                    f"📂 No portfolio loaded for {sel_account}. Upload a statement in the Portfolio tab to see current holdings.")'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Updated to show IBKR live positions")
else:
    print("❌ Could not find the section to replace")
