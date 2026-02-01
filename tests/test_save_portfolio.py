# test_actual_save.py - Test the actual save function

from dashboard.portfolio_builder import get_latest_stock_universe, save_portfolio
from dashboard.portfolio_engine import AIPortfolioEngine, PortfolioIntent

print("=" * 80)
print("TESTING ACTUAL SAVE FUNCTION")
print("=" * 80)

# Build portfolio
df = get_latest_stock_universe(force_refresh=False)
intent = PortfolioIntent(
    objective="biotech_growth",
    max_holdings=5,
    tickers_include=['NVAX', 'NRIX', 'BCRX', 'ALT', 'ORIC']
)

engine = AIPortfolioEngine(df)
result = engine.build_portfolio(intent, "test portfolio")

print(f"\n✓ Built portfolio: {result.num_holdings} holdings")

# Try to save
print("\nAttempting to save portfolio...")
try:
    portfolio_id = save_portfolio("Test Portfolio", result, "Test description")

    if portfolio_id:
        print(f"✅ SUCCESS! Saved with ID: {portfolio_id}")
    else:
        print("✗ FAILED - save_portfolio returned None")

except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback

    print("\nFull traceback:")
    traceback.print_exc()

    # Show exactly which attribute failed
    print("\n" + "=" * 80)
    print("DEBUGGING - Check exact failing attribute:")
    h = result.holdings[0]

    # Try each attribute that save_portfolio uses
    attrs_to_check = [
        ('ticker', lambda: h.ticker),
        ('weight_pct', lambda: h.weight_pct),
        ('shares', lambda: h.shares),
        ('value', lambda: h.value),
        ('sector', lambda: h.sector),
        ('composite_score', lambda: h.composite_score),
        ('signal_type', lambda: h.ai_decision.signal_type if h.ai_decision else None),
        ('rationale', lambda: (h.bull_case[0] if h.bull_case else None) or (
            h.ai_decision.one_line_summary if h.ai_decision else None)),
    ]

    for name, getter in attrs_to_check:
        try:
            value = getter()
            print(f"  ✓ {name}: {value}")
        except Exception as e:
            print(f"  ✗ {name}: ERROR - {e}")

print("\n" + "=" * 80)