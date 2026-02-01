with open('../dashboard/portfolio_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find where Saved Portfolios section starts and add Strategy Builder before it
old_saved = '''        st.markdown("##### 💾 Saved Portfolios")'''

new_strategy_section = '''        # Strategy-Based Portfolio Builder
        st.markdown("##### 🎯 Build by Strategy")
        
        # Define available strategies with display names
        STRATEGY_OPTIONS = {
            "momentum": "📈 Momentum - AI-validated trend following",
            "tech_growth": "💻 Tech Growth - High-growth tech with AI validation", 
            "aggressive_growth": "🚀 Aggressive Growth - Maximum growth with AI signals",
            "biotech_growth": "🧬 Biotech Growth - FDA/clinical catalyst focus",
            "value": "💎 Value - AI-identified undervalued companies",
            "income": "💰 Income - Dividend-focused with AI quality check",
            "balanced": "⚖️ Balanced - AI-optimized multi-factor",
            "conservative": "🛡️ Conservative - Capital preservation",
            "quality": "✨ Quality Factor - High-quality companies",
            "shariah": "☪️ Shariah Compliant - Islamic finance screening",
        }
        
        col_strat1, col_strat2 = st.columns([2, 1])
        
        with col_strat1:
            selected_strategy = st.selectbox(
                "Select Strategy",
                options=list(STRATEGY_OPTIONS.keys()),
                format_func=lambda x: STRATEGY_OPTIONS[x],
                key="pb_strategy_select"
            )
        
        with col_strat2:
            strategy_value = st.number_input(
                "Portfolio Value ($)",
                min_value=10000,
                max_value=10000000,
                value=100000,
                step=10000,
                key="pb_strategy_value"
            )
        
        col_strat3, col_strat4, col_strat5 = st.columns(3)
        
        with col_strat3:
            strategy_risk = st.selectbox(
                "Risk Level",
                options=["conservative", "moderate", "aggressive"],
                index=1,
                key="pb_strategy_risk"
            )
        
        with col_strat4:
            strategy_holdings = st.slider(
                "Max Holdings",
                min_value=10,
                max_value=50,
                value=25,
                key="pb_strategy_holdings"
            )
        
        with col_strat5:
            strategy_max_pos = st.slider(
                "Max Position %",
                min_value=3,
                max_value=15,
                value=8,
                key="pb_strategy_max_pos"
            )
        
        if st.button("🚀 Build Strategy Portfolio", key="pb_build_strategy", type="primary"):
            if ss.portfolio_builder_df is None:
                st.error("Load stock universe first!")
            else:
                with st.spinner(f"Building {STRATEGY_OPTIONS[selected_strategy].split(' - ')[0]} portfolio..."):
                    try:
                        from dashboard.portfolio_engine import PortfolioIntent, PortfolioEngine
                        
                        intent = PortfolioIntent(
                            objective=selected_strategy,
                            risk_level=strategy_risk,
                            portfolio_value=strategy_value,
                            max_holdings=strategy_holdings,
                            max_position_pct=strategy_max_pos,
                            fully_invested=True,
                            equal_weight=False
                        )
                        
                        engine = PortfolioEngine(ss.portfolio_builder_df)
                        result = engine.build_portfolio(intent, user_request=f"{STRATEGY_OPTIONS[selected_strategy]} portfolio")
                        
                        if result and result.success:
                            ss.portfolio_builder_last_result = result
                            ss.last_portfolio_result = result
                            ss.portfolio_builder_loaded_info = None
                            ss.portfolio_builder_loaded_holdings = None
                            ss.portfolio_builder_last_errors = []
                            
                            st.success(f"✅ Built {result.num_holdings} holdings using {STRATEGY_OPTIONS[selected_strategy].split(' - ')[0]}")
                            st.rerun()
                        else:
                            warnings = getattr(result, 'warnings', []) if result else ['Build failed']
                            st.error(f"Failed: {', '.join(str(w) for w in warnings)}")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        st.markdown("---")
        
        st.markdown("##### 💾 Saved Portfolios")'''

if old_saved in content:
    content = content.replace(old_saved, new_strategy_section)
    with open('../dashboard/portfolio_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added Strategy Builder section!")
else:
    print("❌ Could not find Saved Portfolios marker")
