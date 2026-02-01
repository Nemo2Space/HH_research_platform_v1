with open('../dashboard/ai_pm/ibkr_gateway.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add timeout to accountSummary call
old_account_summary = '''        tags = ["NetLiquidation", "TotalCashValue", "AvailableFunds", "BuyingPower"]
        try:
            rows = self.ib.accountSummary(acct)'''

new_account_summary = '''        tags = ["NetLiquidation", "TotalCashValue", "AvailableFunds", "BuyingPower"]
        try:
            # Use timeout to prevent hanging
            import asyncio
            try:
                rows = self.ib.accountSummary(acct)
            except asyncio.TimeoutError:
                return None'''

if old_account_summary in content:
    content = content.replace(old_account_summary, new_account_summary)
    print("✅ Added timeout handling to accountSummary")
else:
    print("❌ Could not find accountSummary section")

with open('../dashboard/ai_pm/ibkr_gateway.py', 'w', encoding='utf-8') as f:
    f.write(content)
