with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the entire JSON import section with enhanced version
old_section = '''        # JSON Import Section
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
                                            
                                            # Store in session state - BOTH keys needed!
                                            ss.portfolio_builder_last_result = result
                                            ss.last_portfolio_result = result  # This triggers enhanced display
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

new_section = '''        # JSON Import Section
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
                    json_names = {item.get('ticker', '').upper(): item.get('name', '') for item in data}
                    
                    st.success(f"✅ Found {len(json_tickers)} tickers in JSON")
                    
                    # Check which tickers are in the database
                    if ss.portfolio_builder_df is not None:
                        available = set(ss.portfolio_builder_df['ticker'].str.upper())
                        found_tickers = [t for t in json_tickers if t in available]
                        missing_tickers = [t for t in json_tickers if t not in available]
                        
                        # Show summary
                        col_info1, col_info2, col_info3 = st.columns(3)
                        col_info1.metric("Total in JSON", len(json_tickers))
                        col_info2.metric("Found in DB", len(found_tickers))
                        col_info3.metric("Missing", len(missing_tickers))
                        
                        if missing_tickers:
                            with st.expander(f"⚠️ {len(missing_tickers)} tickers not in database", expanded=True):
                                missing_df = pd.DataFrame([
                                    {'Ticker': t, 'Name': json_names.get(t, ''), 'Weight': json_weights.get(t, 0)}
                                    for t in missing_tickers
                                ])
                                st.dataframe(missing_df, hide_index=True)
                                
                                if st.button("➕ Add Missing Tickers to Database", key="pb_add_missing"):
                                    import yfinance as yf
                                    import psycopg2
                                    
                                    progress = st.progress(0)
                                    status = st.empty()
                                    added = []
                                    failed = []
                                    
                                    try:
                                        conn = psycopg2.connect(
                                            host='localhost', port=5432, 
                                            dbname='alpha_platform', user='alpha', password='alpha_secure_2024'
                                        )
                                        cur = conn.cursor()
                                        
                                        for idx, ticker in enumerate(missing_tickers):
                                            status.text(f"Fetching {ticker}...")
                                            progress.progress((idx + 1) / len(missing_tickers))
                                            
                                            try:
                                                stock = yf.Ticker(ticker)
                                                info = stock.info
                                                
                                                if info.get('symbol'):
                                                    cur.execute("""
                                                        INSERT INTO fundamentals (ticker, sector, market_cap, pe_ratio, dividend_yield, date)
                                                        VALUES (%s, %s, %s, %s, %s, CURRENT_DATE)
                                                        ON CONFLICT (ticker, date) DO UPDATE SET
                                                            sector = EXCLUDED.sector,
                                                            market_cap = EXCLUDED.market_cap,
                                                            pe_ratio = EXCLUDED.pe_ratio,
                                                            dividend_yield = EXCLUDED.dividend_yield
                                                    """, (
                                                        ticker,
                                                        info.get('sector', 'Unknown'),
                                                        info.get('marketCap'),
                                                        info.get('trailingPE'),
                                                        info.get('dividendYield')
                                                    ))
                                                    added.append(ticker)
                                                else:
                                                    failed.append(f"{ticker}: Not found")
                                            except Exception as fetch_err:
                                                failed.append(f"{ticker}: {str(fetch_err)[:30]}")
                                        
                                        conn.commit()
                                        conn.close()
                                        
                                        status.empty()
                                        progress.empty()
                                        
                                        if added:
                                            st.success(f"✅ Added {len(added)} tickers: {', '.join(added)}")
                                            st.info("🔄 Click 'Refresh' to reload stock universe with new tickers")
                                        if failed:
                                            st.warning(f"⚠️ Failed: {', '.join(failed[:5])}")
                                    except Exception as db_err:
                                        st.error(f"Database error: {db_err}")
                        
                        # Build portfolio buttons
                        if found_tickers:
                            st.markdown("---")
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                build_btn = st.button("🚀 Build Portfolio from JSON", key="pb_build_from_json", type="primary")
                            
                            with col2:
                                # Preview table
                                preview_df = pd.DataFrame([
                                    {'Ticker': t, 'Weight': f"{json_weights.get(t, 0):.1f}%", 'Status': '✅ In DB' if t in found_tickers else '❌ Missing'}
                                    for t in json_tickers
                                ])
                                st.dataframe(preview_df, hide_index=True, height=200)
                            
                            if build_btn:
                                with st.spinner("Building portfolio..."):
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
                                            for h in result.holdings:
                                                ticker_upper = h.ticker.upper()
                                                if ticker_upper in json_weights:
                                                    h.weight_pct = json_weights[ticker_upper]
                                                    h.value = 100000 * h.weight_pct / 100
                                            
                                            # Store in session state - BOTH keys for display
                                            ss.portfolio_builder_last_result = result
                                            ss.last_portfolio_result = result
                                            ss.portfolio_builder_loaded_info = None
                                            ss.portfolio_builder_loaded_holdings = None
                                            ss.portfolio_builder_last_errors = []
                                            
                                            st.success(f"✅ Built portfolio with {result.num_holdings} stocks!")
                                            st.rerun()
                                        else:
                                            warnings = getattr(result, 'warnings', []) if result else ['Build failed']
                                            st.error(f"Failed: {', '.join(warnings)}")
                                    except Exception as build_err:
                                        st.error(f"Build error: {build_err}")
                                        import traceback
                                        st.code(traceback.format_exc())
                        else:
                            st.error("No valid tickers found in database. Add missing tickers first.")
                    else:
                        st.warning("⚠️ Load stock universe first to validate tickers")
                else:
                    st.error("Invalid JSON format - expected array of objects with 'ticker' and 'weight'")
            except Exception as e:
                st.error(f"JSON parse error: {e}")
        
        st.markdown("---")'''

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('../dashboard/portfolio_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Enhanced JSON import with data fetching!")
else:
    print("❌ Could not find exact code block, checking...")
    if '# JSON Import Section' in content:
        print("Found JSON Import Section marker")
    if 'pb_json_upload' in content:
        print("Found json upload key")
