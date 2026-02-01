with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add JSON import section before Saved Portfolios
old_code = '''        st.markdown("##### 💾 Saved Portfolios")'''

new_code = '''        # JSON Import Section
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
        
        st.markdown("---")
        
        st.markdown("##### 💾 Saved Portfolios")'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/portfolio_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added JSON import section to Portfolio Builder!")
else:
    print("❌ Could not find the exact location")
