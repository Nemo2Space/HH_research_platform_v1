"""Quick database summary check."""
import sys
sys.path.insert(0, '.')

from src.db.portfolio_db import PortfolioRepository

repo = PortfolioRepository()

# List accounts
accounts = repo.get_all_accounts()
print("=" * 60)
print("  Accounts in Database")
print("=" * 60)
for acc in accounts:
    print(f"  - {acc['account_id']}: {acc['name']}")

print()

# Get summaries
for acc_id in ['U20993660', 'U17994267']:
    summary = repo.get_account_summary(acc_id)
    if summary:
        nav = summary['nav']
        print(f"\n{acc_id}:")
        if nav:
            print(f"  NAV: ${float(nav['nav_total']):,.2f}")
        else:
            print("  NAV: N/A")
        print(f"  Deposits: ${summary['total_deposits']:,.2f}")
        print(f"  True Profit: ${summary['true_profit']:,.2f} ({summary['true_profit_pct']:.2f}%)")
        print(f"  Dividends: ${summary['total_dividends']:,.2f}")
        print(f"  Interest: ${summary['total_interest']:,.2f}")
        print(f"  Withholding Tax: ${summary['total_withholding_tax']:,.2f}")
    else:
        print(f"\n{acc_id}: Not found")

print("\n" + "=" * 60)