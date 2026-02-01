with open('../dashboard/ai_pm/ibkr_gateway.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The issue is that accountSummary() doesn't return NetLiquidation, but accountValues() does
# Replace the function to use accountValues instead

old_code = '''        try:
            rows = None

            # Prefer async + util.run(timeout=...) if available (true timeout)
            if hasattr(self.ib, "accountSummaryAsync"):
                try:
                    # Try common signatures safely
                    try:
                        rows = util.run(self.ib.accountSummaryAsync(group="All", tags=",".join(tags)),
                                        timeout=timeout_sec)
                    except TypeError:
                        rows = util.run(self.ib.accountSummaryAsync("All", ",".join(tags)), timeout=timeout_sec)
                except Exception:
                    rows = None
            else:
                # Fallback: sync call (still might block if IBKR stalls)
                try:
                    try:
                        rows = self.ib.accountSummary(group="All", tags=",".join(tags))
                    except TypeError:
                        rows = self.ib.accountSummary("All", ",".join(tags))
                except Exception:
                    rows = None

            rows = _filter_rows(rows)'''

new_code = '''        try:
            rows = None

            # Use accountValues() instead of accountSummary() - it has NetLiquidation
            try:
                rows = self.ib.accountValues(acct)
            except Exception:
                rows = None

            rows = _filter_rows(rows)'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('../dashboard/ai_pm/ibkr_gateway.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed get_account_summary to use accountValues()")
else:
    print("❌ Could not find the code to replace")
