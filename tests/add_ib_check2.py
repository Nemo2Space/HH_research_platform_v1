with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace - exact match
old_code = '''about to call execute_trade_plan...")
                        st.info("⏳ Sending orders...")
                    execution = execute_trade_plan(
                        ib=gw.ib,'''

new_code = '''about to call execute_trade_plan...")
                        
                        # Verify IB connection is alive before sending orders
                        if not gw.ib.isConnected():
                            _exec_log.error("Execution: IB not connected!")
                            st.error("❌ IBKR connection lost. Please reconnect.")
                            st.stop()
                        
                        _exec_log.info(f"Execution: IB connected, clientId={gw.ib.client.clientId}")
                        st.info("⏳ Sending orders...")
                    execution = execute_trade_plan(
                        ib=gw.ib,'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added IB connection check before execution")
else:
    print("❌ Could not find - showing context:")
    idx = content.find('about to call execute_trade_plan')
    print(repr(content[idx:idx+300]))
