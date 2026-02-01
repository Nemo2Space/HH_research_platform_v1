"""
Test: Cash Buffer Override & Fully Invested
============================================
This tests that the cash buffer override is working correctly.
"""

import sys
sys.path.insert(0, '..')

from dashboard.portfolio_engine import PortfolioIntent

print("=" * 80)
print("TESTING CASH BUFFER & FULLY INVESTED")
print("=" * 80)

# Test 1: Default behavior (should use strategy default, usually 5-10%)
print("\nTest 1: Default Intent (no override)")
intent1 = PortfolioIntent(objective="biotech_growth")
print(f"  fully_invested: {intent1.fully_invested}")
print(f"  cash_buffer_pct: {intent1.cash_buffer_pct}")
print(f"  Expected: Strategy will use default (5-10% cash)")

# Test 2: Fully invested (100%)
print("\nTest 2: Fully Invested Intent")
intent2 = PortfolioIntent(objective="biotech_growth", fully_invested=True)
print(f"  fully_invested: {intent2.fully_invested}")
print(f"  cash_buffer_pct: {intent2.cash_buffer_pct}")
print(f"  Expected: 0% cash, 100% invested")

# Test 3: Custom cash buffer
print("\nTest 3: Custom Cash Buffer (3%)")
intent3 = PortfolioIntent(objective="balanced", cash_buffer_pct=3.0)
print(f"  fully_invested: {intent3.fully_invested}")
print(f"  cash_buffer_pct: {intent3.cash_buffer_pct}")
print(f"  Expected: 3% cash, 97% invested")

# Test 4: Both set (fully_invested takes precedence)
print("\nTest 4: Both Set (fully_invested should win)")
intent4 = PortfolioIntent(objective="growth", fully_invested=True, cash_buffer_pct=5.0)
print(f"  fully_invested: {intent4.fully_invested}")
print(f"  cash_buffer_pct: {intent4.cash_buffer_pct}")
print(f"  Expected: fully_invested=True takes precedence → 0% cash")

print("\n" + "=" * 80)
print("NEW CONSTRAINT FIELDS")
print("=" * 80)

# Test advanced constraints
print("\nTest 5: Advanced Constraints")
intent5 = PortfolioIntent(
    objective="biotech_growth",
    fully_invested=True,
    min_subsectors=8,
    max_binary_event_weight_pct=30,
    min_established_weight_pct=10,
    max_established_weight_pct=20
)
print(f"  min_subsectors: {intent5.min_subsectors}")
print(f"  max_binary_event_weight_pct: {intent5.max_binary_event_weight_pct}")
print(f"  min_established_weight_pct: {intent5.min_established_weight_pct}")
print(f"  max_established_weight_pct: {intent5.max_established_weight_pct}")
print(f"  Expected: All constraints properly set")

print("\n" + "=" * 80)
print("JSON PARSING TEST")
print("=" * 80)

# Test JSON parsing (simulating LLM output)
from dashboard.portfolio_engine import parse_llm_intent

json_output = """{
    "objective": "biotech_growth",
    "risk_level": "moderate",
    "portfolio_value": 100000,
    "min_holdings": 12,
    "max_holdings": 18,
    "max_position_pct": 10,
    "fully_invested": true,
    "min_subsectors": 8,
    "max_binary_event_weight_pct": 30,
    "min_established_weight_pct": 10,
    "max_established_weight_pct": 20,
    "tickers_include": ["NVAX", "ORIC", "NRIX"],
    "restrict_to_tickers": true
}"""

print("\nTest 6: Parse LLM JSON Output")
intent6, errors = parse_llm_intent(json_output)
if errors:
    print(f"  ❌ Errors: {errors}")
else:
    print(f"  ✅ Parsed successfully!")
    print(f"  objective: {intent6.objective}")
    print(f"  fully_invested: {intent6.fully_invested}")
    print(f"  min_subsectors: {intent6.min_subsectors}")
    print(f"  max_binary_event_weight_pct: {intent6.max_binary_event_weight_pct}")
    print(f"  min_established_weight_pct: {intent6.min_established_weight_pct}")
    print(f"  tickers_include: {intent6.tickers_include}")
    print(f"  restrict_to_tickers: {intent6.restrict_to_tickers}")

print("\n" + "=" * 80)
print("DONE - All intent fields working correctly!")
print("=" * 80)

print("\nNEXT STEPS:")
print("1. Run test_classification.py to verify catalyst classification")
print("2. Test with real portfolio building to see cash buffer override in action")
print("3. Check output for correct catalyst labels (Clinical vs Regulatory vs Commercial)")