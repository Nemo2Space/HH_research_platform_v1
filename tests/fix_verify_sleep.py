with open('../dashboard/ai_pm/ui_tab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the verification code - remove gw.ib.sleep which freezes
old_code = '''                    # Post-execution verification
                    try:
                        from .execution_verify import verify_execution, PortfolioVerification
                        with st.spinner("Verifying orders in TWS..."):
                            # Wait for orders to be visible in TWS
                            import time
                            time.sleep(2)
                            gw.ib.sleep(1)'''

new_code = '''                    # Post-execution verification
                    try:
                        from .execution_verify import verify_execution, PortfolioVerification
                        with st.spinner("Verifying orders in TWS..."):
                            # Wait for orders to be visible in TWS
                            import time
                            time.sleep(3)  # Use regular sleep, not ib.sleep which freezes in Streamlit'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/ui_tab.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed verification - removed ib.sleep")
else:
    print("❌ Could not find code")
