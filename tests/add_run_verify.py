with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the Run Now check and add a real ping test
old_run_check = '''    if run_now:
        if not st.session_state.get('ai_pm_connection_ok', False):
            st.error("❌ IBKR not connected. Click **Reconnect** button above.")
            st.stop()
        elif not gw or not gw.is_connected():
            st.error("❌ IBKR connection lost. Click **Reconnect** button above.")
            st.session_state.ai_pm_connection_ok = False
            st.stop()
        elif not sel_account:
            st.error("❌ Select a single account (not 'All') to run planning.")
            st.stop()'''

new_run_check = '''    if run_now:
        if not st.session_state.get('ai_pm_connection_ok', False):
            st.error("❌ IBKR not connected. Click **Reconnect** button above.")
            st.stop()
        elif not gw or not gw.is_connected():
            st.error("❌ IBKR connection lost. Click **Reconnect** button above.")
            st.session_state.ai_pm_connection_ok = False
            st.stop()
        elif not sel_account:
            st.error("❌ Select a single account (not 'All') to run planning.")
            st.stop()
        
        # Verify connection is truly alive with a quick test
        try:
            test_accounts = gw.list_accounts()
            if not test_accounts:
                st.error("❌ IBKR connection stale. Click **Reconnect**.")
                st.session_state.ai_pm_connection_ok = False
                st.stop()
        except Exception as e:
            st.error(f"❌ IBKR connection error: {e}. Click **Reconnect**.")
            st.session_state.ai_pm_connection_ok = False
            st.stop()'''

if old_run_check in content:
    content = content.replace(old_run_check, new_run_check)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added connection verification to Run Now")
else:
    print("❌ Could not find run check")
