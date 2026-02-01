with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find where to add the saved portfolio section - after the max holdings slider
# Look for the Auto Trade toggle which comes after the sliders

old_section = '''    with right:
        c1, c2 = st.columns(2)
        with c1:
            auto = st.toggle("Auto Trade", value=is_auto_trade(), key="ai_pm_auto_trade_toggle")
            set_auto_trade(auto)
        with c2:
            kill = st.toggle("Kill Switch", value=is_kill_switch(), key="ai_pm_kill_switch_toggle")
            set_kill_switch(kill)'''

new_section = '''    with right:
        c1, c2 = st.columns(2)
        with c1:
            auto = st.toggle("Auto Trade", value=is_auto_trade(), key="ai_pm_auto_trade_toggle")
            set_auto_trade(auto)
        with c2:
            kill = st.toggle("Kill Switch", value=is_kill_switch(), key="ai_pm_kill_switch_toggle")
            set_kill_switch(kill)

    # ==========================================================================
    # LOAD SAVED PORTFOLIO AS TARGET
    # ==========================================================================
    if SAVED_PORTFOLIO_AVAILABLE:
        st.markdown("---")
        st.markdown("##### 📂 Load Saved Portfolio as Target")
        
        # Initialize session state for saved portfolio
        if 'ai_pm_target_portfolio' not in st.session_state:
            st.session_state.ai_pm_target_portfolio = None
        if 'ai_pm_target_weights' not in st.session_state:
            st.session_state.ai_pm_target_weights = {}
        
        saved_portfolios_df = get_saved_portfolios()
        
        if saved_portfolios_df is not None and not saved_portfolios_df.empty:
            col_port1, col_port2, col_port3 = st.columns([3, 1, 1])
            
            with col_port1:
                # Create display names with info
                portfolio_options = ["-- Select Saved Portfolio --"] + [
                    f"{row['name']} ({row['num_holdings']} stocks, )"
                    for _, row in saved_portfolios_df.iterrows()
                ]
                portfolio_ids = [None] + saved_portfolios_df['id'].tolist()
                
                selected_idx = st.selectbox(
                    "Saved Portfolio",
                    options=range(len(portfolio_options)),
                    format_func=lambda i: portfolio_options[i],
                    key="ai_pm_saved_portfolio_select",
                    label_visibility="collapsed"
                )
            
            with col_port2:
                load_btn = st.button("📥 Load as Target", key="ai_pm_load_portfolio_btn", type="primary")
            
            with col_port3:
                clear_btn = st.button("🗑️ Clear Target", key="ai_pm_clear_target_btn")
            
            if load_btn and selected_idx and selected_idx > 0:
                portfolio_id = portfolio_ids[selected_idx]
                info, holdings_df = load_portfolio(portfolio_id)
                
                if info and holdings_df is not None and not holdings_df.empty:
                    # Convert holdings to target weights dict
                    target_weights = {}
                    for _, row in holdings_df.iterrows():
                        ticker = row.get('ticker', '').upper().strip()
                        weight = float(row.get('weight_pct', 0) or 0) / 100  # Convert to decimal
                        if ticker and weight > 0:
                            target_weights[ticker] = weight
                    
                    # Store in session state
                    st.session_state.ai_pm_target_portfolio = {
                        'id': portfolio_id,
                        'name': info.get('name', 'Unknown'),
                        'total_value': info.get('total_value', 100000),
                        'num_holdings': len(target_weights),
                        'objective': info.get('objective', 'custom'),
                    }
                    st.session_state.ai_pm_target_weights = target_weights
                    
                    st.success(f"✅ Loaded **{info.get('name')}** with {len(target_weights)} holdings as target")
                    st.rerun()
                else:
                    st.error("Failed to load portfolio")
            
            if clear_btn:
                st.session_state.ai_pm_target_portfolio = None
                st.session_state.ai_pm_target_weights = {}
                st.info("Target portfolio cleared")
                st.rerun()
            
            # Show current target if loaded
            if st.session_state.ai_pm_target_portfolio:
                tp = st.session_state.ai_pm_target_portfolio
                tw = st.session_state.ai_pm_target_weights
                
                st.info(f"🎯 **Active Target:** {tp['name']} | {tp['num_holdings']} holdings | ")
                
                with st.expander("View Target Holdings", expanded=False):
                    target_df = pd.DataFrame([
                        {'Ticker': k, 'Weight %': f"{v*100:.2f}%"}
                        for k, v in sorted(tw.items(), key=lambda x: -x[1])
                    ])
                    st.dataframe(target_df, hide_index=True, height=300)
        else:
            st.caption("No saved portfolios found. Create one in the AI Portfolio Builder tab.")'''

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added Saved Portfolio section to AI PM!")
else:
    print("❌ Could not find the section to replace")
    # Debug
    if 'Auto Trade' in content:
        print("Found Auto Trade toggle")
    if 'Kill Switch' in content:
        print("Found Kill Switch toggle")
