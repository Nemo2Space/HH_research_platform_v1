with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the first build_target_weights call
old_code1 = '''                # Get max holdings from session state (user-controlled slider)
                max_holdings_override = st.session_state.get('ai_pm_max_holdings')

                targets = build_target_weights(
                    signals=signals,
                    strategy_key=get_strategy_key(),
                    constraints=DEFAULT_CONSTRAINTS,
                    max_holdings_override=max_holdings_override,
                )

                # NEW: Show target diagnostics with warnings
                target_count = len(targets.weights) if targets.weights else 0
                st.info(
                    f"🎯 **Strategy:** {get_strategy_key()} | **Max Holdings:** {max_holdings_override or 'default'} → **{target_count} holdings** selected")'''

new_code1 = '''                # Get max holdings from session state (user-controlled slider)
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
                    st.success(f"📂 Using saved portfolio **{tp.get('name')}** as target ({len(targets.weights)} holdings)")
                else:
                    targets = build_target_weights(
                        signals=signals,
                        strategy_key=get_strategy_key(),
                        constraints=DEFAULT_CONSTRAINTS,
                        max_holdings_override=max_holdings_override,
                    )

                # NEW: Show target diagnostics with warnings
                target_count = len(targets.weights) if targets.weights else 0
                st.info(
                    f"🎯 **Strategy:** {get_strategy_key()} | **Max Holdings:** {max_holdings_override or 'default'} → **{target_count} holdings** selected")'''

if old_code1 in content:
    content = content.replace(old_code1, new_code1)
    print("✅ Updated first build_target_weights call")
else:
    print("❌ Could not find first target weights section")

with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
    f.write(content)
