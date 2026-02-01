with open('../dashboard/ai_pm/ibkr_gateway.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_open_orders = '''    def get_open_orders(self, include_all_clients: bool = True) -> List[OpenOrderRow]:
        """
        Returns open orders.
        If include_all_clients=True, asks TWS for all clients' orders (useful if multiple scripts).
        """
        if not self.is_connected():
            return []

        try:
            if include_all_clients:
                self.ib.reqAllOpenOrders()
                # small settle time for TWS stream
                time.sleep(0.3)

            trades = list(self.ib.openTrades() or [])'''

new_open_orders = '''    def get_open_orders(self, include_all_clients: bool = True) -> List[OpenOrderRow]:
        """
        Returns open orders.
        If include_all_clients=True, asks TWS for all clients' orders (useful if multiple scripts).
        NOTE: reqAllOpenOrders() can hang in Streamlit, so we skip it and just use openTrades().
        """
        if not self.is_connected():
            return []

        try:
            # SKIP reqAllOpenOrders() - it hangs in Streamlit's event loop
            # Just use openTrades() which has the orders from our connection
            trades = list(self.ib.openTrades() or [])'''

if old_open_orders in content:
    content = content.replace(old_open_orders, new_open_orders)
    with open('../dashboard/ai_pm/ibkr_gateway.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed get_open_orders - removed reqAllOpenOrders() that hangs")
else:
    print("❌ Could not find get_open_orders")
