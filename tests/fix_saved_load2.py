with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the saved portfolio loading with a simpler approach that works
old_code = '''                    if st.button(f"📂 {pname}", key=f"pb_load_{pid}"):
                        info, holdings = load_portfolio(pid)
                        if info is None or holdings is None:
                            st.error("Load failed.")
                        else:
                            ss.portfolio_builder_loaded_info = info
                            ss.portfolio_builder_loaded_holdings = holdings
                            ss.portfolio_builder_last_errors = []
                            
                            # Convert to PortfolioResult for enhanced display
                            try:
                                from dashboard.portfolio_engine import PortfolioResult, PortfolioHolding
                                
                                # Reconstruct holdings from DataFrame
                                result_holdings = []
                                for _, row in holdings.iterrows():
                                    h = PortfolioHolding(
                                        ticker=row.get('ticker', ''),
                                        sector=row.get('sector', 'Unknown'),
                                        weight_pct=float(row.get('weight_pct', 0) or 0),
                                        value=float(row.get('value', 0) or 0),
                                        shares=float(row.get('shares', 0) or 0),
                                        price=float(row.get('price', 0) or 0),
                                        market_cap=float(row.get('market_cap', 0) or 0),
                                        composite_score=float(row.get('composite_score', 50) or 50),
                                        conviction=row.get('conviction', 'MEDIUM'),
                                        ai_action=row.get('ai_action'),
                                        ai_probability=float(row.get('ai_probability', 0) or 0) if row.get('ai_probability') else None,
                                        signal_type=row.get('signal_type'),
                                        signal_strength=int(row.get('signal_strength', 0) or 0) if row.get('signal_strength') else None
                                    )
                                    result_holdings.append(h)
                                
                                result = PortfolioResult(
                                    success=True,
                                    holdings=result_holdings,
                                    total_value=float(info.get('total_value', 100000) or 100000),
                                    cash_value=float(info.get('cash_value', 0) or 0),
                                    invested_value=float(info.get('invested_value', 100000) or 100000),
                                    num_holdings=len(result_holdings),
                                    objective=info.get('objective', 'custom'),
                                    risk_level=info.get('risk_level', 'moderate'),
                                    warnings=[]
                                )
                                
                                ss.portfolio_builder_last_result = result
                                ss.last_portfolio_result = result  # Triggers enhanced display
                            except Exception as conv_err:
                                import logging
                                logging.error(f"Could not convert saved portfolio for display: {conv_err}")
                                ss.portfolio_builder_last_result = None
                                ss.last_portfolio_result = None
                            
                            st.rerun()'''

new_code = '''                    if st.button(f"📂 {pname}", key=f"pb_load_{pid}"):
                        info, holdings = load_portfolio(pid)
                        if info is None or holdings is None:
                            st.error("Load failed.")
                        else:
                            ss.portfolio_builder_loaded_info = info
                            ss.portfolio_builder_loaded_holdings = holdings
                            ss.portfolio_builder_last_errors = []
                            
                            # Re-build portfolio using engine for proper display
                            try:
                                from dashboard.portfolio_engine import PortfolioIntent, PortfolioEngine
                                
                                # Get tickers from saved holdings
                                saved_tickers = holdings['ticker'].tolist() if 'ticker' in holdings.columns else []
                                saved_weights = dict(zip(holdings['ticker'], holdings['weight_pct'])) if 'ticker' in holdings.columns and 'weight_pct' in holdings.columns else {}
                                
                                if saved_tickers and ss.portfolio_builder_df is not None:
                                    intent = PortfolioIntent(
                                        objective=info.get('objective', 'custom'),
                                        risk_level=info.get('risk_level', 'moderate'),
                                        portfolio_value=float(info.get('total_value', 100000) or 100000),
                                        tickers_include=saved_tickers,
                                        restrict_to_tickers=True,
                                        fully_invested=True,
                                        equal_weight=False,
                                        max_position_pct=15
                                    )
                                    
                                    engine = PortfolioEngine(ss.portfolio_builder_df)
                                    result = engine.build_portfolio(intent, user_request=f"Loaded: {pname}")
                                    
                                    if result and result.success:
                                        # Apply saved weights
                                        for h in result.holdings:
                                            if h.ticker in saved_weights:
                                                h.weight_pct = float(saved_weights[h.ticker])
                                                h.value = float(info.get('total_value', 100000) or 100000) * h.weight_pct / 100
                                        
                                        ss.portfolio_builder_last_result = result
                                        ss.last_portfolio_result = result
                                    else:
                                        ss.portfolio_builder_last_result = None
                                        ss.last_portfolio_result = None
                                else:
                                    ss.portfolio_builder_last_result = None
                                    ss.last_portfolio_result = None
                            except Exception as conv_err:
                                import logging
                                logging.error(f"Could not rebuild saved portfolio: {conv_err}")
                                ss.portfolio_builder_last_result = None
                                ss.last_portfolio_result = None
                            
                            st.rerun()'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/portfolio_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed saved portfolio loading!")
else:
    print("❌ Could not find the code block")
