with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the section after gates evaluation and add detailed logging
old_section = '''                _exec_log.info(f"Execution: gates evaluated, blocked={gates.blocked}")

                if gates.blocked:
                    st.error("Blocked by hard gates; not sending orders.")
                elif is_kill_switch():
                    st.error("Kill switch enabled; not sending orders.")
                else:
                    _exec_log.info("Execution: about to call execute_trade_plan...")
                    st.info("⏳ Sending orders...")'''

new_section = '''                _exec_log.info(f"Execution: gates evaluated, blocked={gates.blocked}")

                _exec_log.info("Execution: checking gates.blocked...")
                if gates.blocked:
                    _exec_log.info("Execution: gates.blocked is True, showing error")
                    st.error("Blocked by hard gates; not sending orders.")
                else:
                    _exec_log.info("Execution: gates.blocked is False, checking kill switch...")
                    kill_sw = is_kill_switch()
                    _exec_log.info(f"Execution: is_kill_switch() = {kill_sw}")
                    if kill_sw:
                        st.error("Kill switch enabled; not sending orders.")
                    else:
                        _exec_log.info("Execution: about to call execute_trade_plan...")
                        st.info("⏳ Sending orders...")'''

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added detailed logging around gates check")
else:
    print("❌ Could not find section - checking what's there")
    idx = content.find('gates evaluated, blocked=')
    if idx > 0:
        print(repr(content[idx:idx+600]))
