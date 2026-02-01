with open('../dashboard/ai_pm/ibkr_gateway.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace get_account_summary with heavily debugged version
old_func = '''    def get_account_summary(self, account: Optional[str] = None) -> Optional[AccountSummary]:
        """
        Returns key account summary fields (single account only).
        If account is None, uses selected account; if still None, picks first managed account.
        """
        if not self.is_connected():
            return None

        acct = account or self._selected_account
        if not acct:
            accts = self.list_accounts()
            acct = accts[0] if accts else None
        if not acct:
            return None

        tags = ["NetLiquidation", "TotalCashValue", "AvailableFunds", "BuyingPower"]
        try:
            # Use timeout to prevent hanging
            import asyncio
            try:
                rows = self.ib.accountSummary(acct)
            except asyncio.TimeoutError:
                return None'''

new_func = '''    def get_account_summary(self, account: Optional[str] = None) -> Optional[AccountSummary]:
        """
        Returns key account summary fields (single account only).
        If account is None, uses selected account; if still None, picks first managed account.
        """
        import logging
        import sys
        _log = logging.getLogger(__name__)
        
        _log.info("get_account_summary: START")
        print("DEBUG get_account_summary: START", flush=True)
        sys.stdout.flush()
        
        if not self.is_connected():
            _log.warning("get_account_summary: NOT CONNECTED")
            print("DEBUG get_account_summary: NOT CONNECTED", flush=True)
            return None

        _log.info("get_account_summary: connected OK")
        print("DEBUG get_account_summary: connected OK", flush=True)
        
        acct = account or self._selected_account
        if not acct:
            _log.info("get_account_summary: getting accounts list...")
            print("DEBUG get_account_summary: getting accounts list...", flush=True)
            accts = self.list_accounts()
            acct = accts[0] if accts else None
        if not acct:
            _log.warning("get_account_summary: NO ACCOUNT")
            print("DEBUG get_account_summary: NO ACCOUNT", flush=True)
            return None

        _log.info(f"get_account_summary: using account {acct}")
        print(f"DEBUG get_account_summary: using account {acct}", flush=True)
        
        tags = ["NetLiquidation", "TotalCashValue", "AvailableFunds", "BuyingPower"]
        try:
            _log.info("get_account_summary: CALLING ib.accountSummary() - THIS MAY HANG")
            print("DEBUG get_account_summary: CALLING ib.accountSummary() - THIS MAY HANG", flush=True)
            sys.stdout.flush()
            
            # Try with explicit timeout
            import asyncio
            loop = None
            try:
                loop = asyncio.get_event_loop()
                _log.info(f"get_account_summary: event loop = {loop}, running = {loop.is_running()}")
                print(f"DEBUG get_account_summary: event loop = {loop}, running = {loop.is_running()}", flush=True)
            except Exception as e:
                _log.warning(f"get_account_summary: no event loop: {e}")
                print(f"DEBUG get_account_summary: no event loop: {e}", flush=True)
            
            rows = self.ib.accountSummary(acct)
            
            _log.info(f"get_account_summary: GOT {len(rows) if rows else 0} rows")
            print(f"DEBUG get_account_summary: GOT {len(rows) if rows else 0} rows", flush=True)'''

if old_func in content:
    content = content.replace(old_func, new_func)
    with open('../dashboard/ai_pm/ibkr_gateway.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added deep debugging to get_account_summary")
else:
    print("❌ Could not find get_account_summary function")
    # Show what we have
    if 'def get_account_summary' in content:
        idx = content.find('def get_account_summary')
        print("Found at:", idx)
        print(content[idx:idx+500])
