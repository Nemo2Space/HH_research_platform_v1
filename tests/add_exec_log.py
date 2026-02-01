with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add logging before execute_trade_plan call
old_code = '''                if gates.blocked:
                    st.error("Blocked by hard gates; not sending orders.")
                elif is_kill_switch():
                    st.error("Kill switch enabled; not sending orders.")
                else:
                    execution = execute_trade_plan('''

new_code = '''                if gates.blocked:
                    st.error("Blocked by hard gates; not sending orders.")
                elif is_kill_switch():
                    st.error("Kill switch enabled; not sending orders.")
                else:
                    _exec_log.info("Execution: about to call execute_trade_plan...")
                    st.info("⏳ Sending orders...")
                    execution = execute_trade_plan('''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added debug before execute_trade_plan")
else:
    print("❌ Could not find code block")
