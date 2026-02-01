with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the second build_target_weights call
old_code2 = '''                signals, diag = load_platform_signals_snapshot(limit=5000)
                max_holdings_override = st.session_state.get('ai_pm_max_holdings')
                targets = build_target_weights(
                    signals=signals,
                    strategy_key=get_strategy_key(),
                    constraints=DEFAULT_CONSTRAINTS,
                    max_holdings_override=max_holdings_override,
                )

                universe = sorted('''

new_code2 = '''                signals, diag = load_platform_signals_snapshot(limit=5000)
                max_holdings_override = st.session_state.get('ai_pm_max_holdings')
                
                # Check if we have a saved portfolio loaded as target
                if st.session_state.get('ai_pm_target_weights'):
                    # Use saved portfolio weights directly
                    from .models import TargetWeights
                    tp = st.session_state.get('ai_pm_target_portfolio', {})
                    targets = TargetWeights(
                        weights=st.session_state['ai_pm_target_weights'],
                        strategy_key='saved_portfolio',
                        generated_at=datetime.now(),
                        notes=[f"Using saved portfolio: {tp.get('name', 'Unknown')}"]
                    )
                else:
                    targets = build_target_weights(
                        signals=signals,
                        strategy_key=get_strategy_key(),
                        constraints=DEFAULT_CONSTRAINTS,
                        max_holdings_override=max_holdings_override,
                    )

                universe = sorted('''

if old_code2 in content:
    content = content.replace(old_code2, new_code2)
    print("✅ Updated second build_target_weights call")
else:
    print("❌ Could not find second target weights section")

with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
    f.write(content)
