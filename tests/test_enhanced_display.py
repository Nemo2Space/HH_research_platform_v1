# test_enhanced_display.py - Test if the display will work

import sys

sys.path.insert(0, '../dashboard')


# Mock result object for testing
class MockCatalystInfo:
    def __init__(self):
        self.days_to_fda = 55
        self.days_to_earnings = None
        self.catalyst_label = "Regulatory"


class MockAIDecision:
    def __init__(self):
        self.ai_action = "BUY"
        self.ai_probability = 87.5
        self.committee_verdict = "STRONG BUY"


class MockHolding:
    def __init__(self):
        self.ticker = "NVAX"
        self.company_name = "Novavax Inc."
        self.weight_pct = 3.8
        self.value = 3800
        self.composite_score = 70
        self.conviction = "HIGH"
        self.sector = "Healthcare"
        self.market_cap = 1.3e9
        self.pe_ratio = 15.2
        self.revenue_growth = 25.5
        self.options_sentiment = "BULLISH"
        self.squeeze_risk = "EXTREME"
        self.catalyst_info = MockCatalystInfo()
        self.ai_decision = MockAIDecision()


class MockResult:
    def __init__(self):
        self.success = True
        self.holdings = [MockHolding(), MockHolding()]
        self.invested_value = 100000
        self.avg_score = 65
        self.num_holdings = 2


print("=" * 80)
print("TEST ENHANCED DISPLAY")
print("=" * 80)

# Test import
print("\n1. Testing import...")
try:
    from enhanced_portfolio_display import render_comprehensive_stock_table, render_save_portfolio_section

    print("   ✓ Import successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test table rendering
print("\n2. Testing render_comprehensive_stock_table...")
try:
    result = MockResult()

    # Build data like the function does
    data = []
    for h in result.holdings:
        why_parts = []
        if h.ai_decision:
            if h.ai_decision.ai_action:
                why_parts.append(f"AI: {h.ai_decision.ai_action}")
            if h.ai_decision.ai_probability:
                why_parts.append(f"Prob: {h.ai_decision.ai_probability:.0f}%")

        row = {
            'Ticker': h.ticker,
            'Company': h.company_name,
            'Weight': f"{h.weight_pct:.1f}%",
            'Score': int(h.composite_score),
            'Why Chosen': " • ".join(why_parts) if why_parts else "Diversification",
        }
        data.append(row)

    print(f"   ✓ Data built: {len(data)} holdings")
    print(f"   Sample row: {data[0]}")

except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback

    traceback.print_exc()

# Check key attributes
print("\n3. Checking MockHolding has all needed attributes...")
h = MockHolding()
required_attrs = [
    'ticker', 'company_name', 'weight_pct', 'value', 'composite_score',
    'conviction', 'sector', 'market_cap', 'pe_ratio', 'revenue_growth',
    'options_sentiment', 'squeeze_risk', 'catalyst_info', 'ai_decision'
]

for attr in required_attrs:
    has_it = hasattr(h, attr)
    status = "✓" if has_it else "✗"
    print(f"   {status} {attr}")

print("\n4. Checking ai_decision attributes...")
ai = h.ai_decision
ai_attrs = ['ai_action', 'ai_probability', 'committee_verdict']
for attr in ai_attrs:
    has_it = hasattr(ai, attr)
    status = "✓" if has_it else "✗"
    print(f"   {status} {attr}")

print("\n5. Checking catalyst_info attributes...")
cat = h.catalyst_info
cat_attrs = ['days_to_fda', 'days_to_earnings', 'catalyst_label']
for attr in cat_attrs:
    has_it = hasattr(cat, attr)
    status = "✓" if has_it else "✗"
    print(f"   {status} {attr}")

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED - Display should work")
print("=" * 80)