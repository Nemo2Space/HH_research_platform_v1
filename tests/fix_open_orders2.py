with open('../dashboard/ai_pm/ibkr_gateway.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the problematic section - match exactly what's there
old_section = '''if include_all_clients:
                self.ib.reqAllOpenOrders()
                # small settle time for TWS stream
                time.sleep(0.3)

            trades = list(self.ib.openTrades() or [])'''

new_section = '''# SKIP reqAllOpenOrders() - it hangs in Streamlit's event loop
            # Just use openTrades() directly
            trades = list(self.ib.openTrades() or [])'''

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('../dashboard/ai_pm/ibkr_gateway.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed get_open_orders - removed reqAllOpenOrders()")
else:
    print("❌ Could not find section")
    # Try to show what's actually there
    idx = content.find('reqAllOpenOrders')
    if idx > 0:
        print(f"Found reqAllOpenOrders at {idx}")
        print(repr(content[idx-50:idx+100]))
