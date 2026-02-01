with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the first occurrence
old1 = '''                    targets = TargetWeights(
                        weights=st.session_state['ai_pm_target_weights'],
                        strategy_key='saved_portfolio',
                        generated_at=datetime.now(),
                        notes=[f"Using saved portfolio: {tp.get('name', 'Unknown')}"]
                    )'''

new1 = '''                    targets = TargetWeights(
                        weights=st.session_state['ai_pm_target_weights'],
                        strategy_key='saved_portfolio',
                        ts_utc=datetime.utcnow(),
                        notes=[f"Using saved portfolio: {tp.get('name', 'Unknown')}"]
                    )'''

# Fix the second occurrence
old2 = '''                    targets = TargetWeights(
                        weights=st.session_state['ai_pm_target_weights'],
                        strategy_key='saved_portfolio',
                        generated_at=datetime.now(),
                        notes=[f"Using saved portfolio: {tp.get('name', 'Unknown')}"]
                    )
                else:'''

new2 = '''                    targets = TargetWeights(
                        weights=st.session_state['ai_pm_target_weights'],
                        strategy_key='saved_portfolio',
                        ts_utc=datetime.utcnow(),
                        notes=[f"Using saved portfolio: {tp.get('name', 'Unknown')}"]
                    )
                else:'''

content = content.replace(old1, new1)
content = content.replace(old2, new2)

with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed TargetWeights - changed generated_at to ts_utc")
