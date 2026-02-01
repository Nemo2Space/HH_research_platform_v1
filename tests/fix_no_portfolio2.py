with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace just the else block
old_else = '''        else:
            st.info(
                f"📂 No portfolio loaded for {sel_account}. Upload a statement in the Portfolio tab to see current holdings.")'''

new_else = '''        else:
            # Check for IBKR live positions when no statement is uploaded
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
                    f"📂 No portfolio loaded for {sel_account}. Connect to IBKR or upload a statement.")'''

if old_else in content:
    content = content.replace(old_else, new_else)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Updated to show IBKR live positions")
else:
    print("❌ Could not find the exact else block")
    # Try to see what's there
    if "No portfolio loaded for" in content:
        print("Found 'No portfolio loaded for' text")
