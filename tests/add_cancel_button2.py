with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_confirm = '''    confirm = st.button(
        "Confirm & Send (Manual)",
        width='stretch',
        disabled=(not can_trade or blocked or is_kill_switch()),
        key="ai_pm_confirm_send_btn",
    )'''

new_confirm = '''    col_confirm, col_cancel = st.columns([1, 1])
    with col_confirm:
        confirm = st.button(
            "Confirm & Send (Manual)",
            width='stretch',
            disabled=(not can_trade or blocked or is_kill_switch()),
            key="ai_pm_confirm_send_btn",
        )
    with col_cancel:
        cancel_all = st.button(
            "🛑 Cancel All Orders",
            width='stretch',
            type="secondary",
            disabled=not gw or not gw.is_connected(),
            key="ai_pm_cancel_all_btn",
        )
        if cancel_all:
            with st.spinner("Cancelling all orders..."):
                try:
                    gw.ib.reqGlobalCancel()
                    gw.ib.sleep(3)
                    st.success("✅ Global cancel request sent")
                except Exception as e:
                    st.error(f"Cancel failed: {e}")'''

if old_confirm in content:
    content = content.replace(old_confirm, new_confirm)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added Cancel All Orders button")
else:
    print("❌ Could not find confirm button section")
