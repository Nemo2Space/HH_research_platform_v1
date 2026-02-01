with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the JSON import section with a fixed version
old_code = '''        # JSON Import Section
        st.markdown("##### 📁 Import Portfolio from JSON")
        uploaded_json = st.file_uploader("Upload JSON file", type=['json'], key="pb_json_upload")
        
        if uploaded_json is not None:
            try:
                import json
                data = json.load(uploaded_json)
                
                if isinstance(data, list) and len(data) > 0:
                    # Parse JSON - extract tickers and weights
                    tickers = [item.get('ticker', '').upper() for item in data if item.get('ticker')]
                    weights = {item.get('ticker', '').upper(): item.get('weight', 0) for item in data}
                    
                    st.info(f"Found {len(tickers)} tickers in JSON")
                    
                    # Check which tickers are in the database
                    if ss.portfolio_builder_df is not None:
                        available = set(ss.portfolio_builder_df['ticker'].str.upper())
                        found = [t for t in tickers if t in available]
                        missing = [t for t in tickers if t not in available]
                        
                        if missing:
                            st.warning(f"⚠️ {len(missing)} tickers not in database: {', '.join(missing[:10])}")
                        
                        if found:
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("🚀 Build Portfolio from JSON", key="pb_build_from_json"):
                                    # Create a prompt with the tickers
                                    ticker_str = ", ".join(found)
                                    total_weight = sum(weights.get(t, 0) for t in found)
                                    
                                    # Build portfolio using the engine directly
                                    from portfolio_engine import PortfolioIntent, PortfolioEngine
                                    
                                    intent = PortfolioIntent(
                                        objective='custom',
                                        risk_level='moderate',
                                        portfolio_value=100000,
                                        tickers_include=found,
                                        restrict_to_tickers=True,
                                        fully_invested=True,
                                        equal_weight=False,
                                        max_position_pct=15
                                    )
                                    
                                    engine = PortfolioEngine(ss.portfolio_builder_df)
                                    result = engine.build_portfolio(intent, user_request=f"Portfolio from JSON with {len(found)} stocks")
                                    
                                    if result.success:
                                        # Apply custom weights from JSON
                                        for h in result.holdings:
                                            if h.ticker.upper() in weights:
                                                h.weight_pct = weights[h.ticker.upper()]
                                                h.value = 100000 * h.weight_pct / 100
                                        
                                        ss.portfolio_builder_last_result = result
                                        ss.portfolio_builder_loaded_info = None
                                        ss.portfolio_builder_loaded_holdings = None
                                        st.success(f"✅ Built portfolio with {result.num_holdings} stocks")
                                        st.rerun()
                                    else:
                                        st.error("Failed to build portfolio")
                            
                            with col2:
                                # Preview the JSON
                                preview_df = pd.DataFrame([
                                    {'Ticker': t, 'Weight': weights.get(t, 0), 'In DB': '✅' if t in found else '❌'}
                                    for t in tickers[:10]
                                ])
                                st.dataframe(preview_df, hide_index=True, height=200)
                    else:
                        st.warning("Load stock universe first to validate tickers")
                else:
                    st.error("Invalid JSON format - expected array of objects with 'ticker' and 'weight'")
            except Exception as e:
                st.error(f"JSON parse error: {e}")
        
        st.markdown("---")'''

new_code = '''        # JSON Import Section
        st.markdown("##### 📁 Import Portfolio from JSON")
        uploaded_json = st.file_uploader("Upload JSON file", type=['json'], key="pb_json_upload")
        
        if uploaded_json is not None:
            try:
                import json as json_lib
                data = json_lib.load(uploaded_json)
                
                if isinstance(data, list) and len(data) > 0:
                    # Parse JSON - extract tickers and weights
                    json_tickers = [item.get('ticker', '').upper() for item in data if item.get('ticker')]
                    json_weights = {item.get('ticker', '').upper(): item.get('weight', 0) for item in data}
                    
                    st.info(f"Found {len(json_tickers)} tickers in JSON")
                    
                    # Check which tickers are in the database
                    if ss.portfolio_builder_df is not None:
                        available = set(ss.portfolio_builder_df['ticker'].str.upper())
                        found_tickers = [t for t in json_tickers if t in available]
                        missing_tickers = [t for t in json_tickers if t not in available]
                        
                        if missing_tickers:
                            st.warning(f"⚠️ {len(missing_tickers)} tickers not in database: {', '.join(missing_tickers[:10])}")
                        
                        if found_tickers:
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("🚀 Build Portfolio from JSON", key="pb_build_from_json"):
                                    try:
                                        from dashboard.portfolio_engine import PortfolioIntent, PortfolioEngine
                                        
                                        intent = PortfolioIntent(
                                            objective='custom',
                                            risk_level='moderate',
                                            portfolio_value=100000,
                                            tickers_include=found_tickers,
                                            restrict_to_tickers=True,
                                            fully_invested=True,
                                            equal_weight=False,
                                            max_position_pct=15
                                        )
                                        
                                        engine = PortfolioEngine(ss.portfolio_builder_df)
                                        result = engine.build_portfolio(intent, user_request=f"Custom Portfolio from JSON ({len(found_tickers)} stocks)")
                                        
                                        if result and result.success:
                                            # Apply custom weights from JSON
                                            total_json_weight = sum(json_weights.get(t, 0) for t in found_tickers)
                                            for h in result.holdings:
                                                ticker_upper = h.ticker.upper()
                                                if ticker_upper in json_weights:
                                                    h.weight_pct = json_weights[ticker_upper]
                                                    h.value = 100000 * h.weight_pct / 100
                                            
                                            # Store in session state
                                            ss.portfolio_builder_last_result = result
                                            ss.portfolio_builder_loaded_info = None
                                            ss.portfolio_builder_loaded_holdings = None
                                            ss.portfolio_builder_last_errors = []
                                            
                                            # Add to chat history for display
                                            ss.portfolio_chat_history.append({
                                                "role": "user",
                                                "content": f"Build portfolio from JSON with {len(found_tickers)} stocks: {', '.join(found_tickers[:10])}..."
                                            })
                                            ss.portfolio_chat_history.append({
                                                "role": "assistant", 
                                                "content": f"json_portfolio_built"
                                            })
                                            
                                            st.success(f"✅ Built portfolio with {result.num_holdings} stocks!")
                                            st.rerun()
                                        else:
                                            st.error(f"Failed to build portfolio: {getattr(result, 'warnings', ['Unknown error'])}")
                                    except Exception as build_err:
                                        st.error(f"Build error: {build_err}")
                            
                            with col2:
                                # Preview the JSON
                                preview_df = pd.DataFrame([
                                    {'Ticker': t, 'Weight': json_weights.get(t, 0), 'In DB': '✅' if t in found_tickers else '❌'}
                                    for t in json_tickers[:10]
                                ])
                                st.dataframe(preview_df, hide_index=True, height=200)
                    else:
                        st.warning("Load stock universe first to validate tickers")
                else:
                    st.error("Invalid JSON format - expected array of objects with 'ticker' and 'weight'")
            except Exception as e:
                st.error(f"JSON parse error: {e}")
        
        st.markdown("---")'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/portfolio_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed JSON import with better error handling!")
else:
    print("❌ Could not find the code block to replace")
