"""
Quick Test: Verify Catalyst Classification
==========================================
This tests that the new classification system is working correctly.
"""

import sys

sys.path.insert(0, '..')

# Test the classification function directly
from dashboard.portfolio_engine import AIPortfolioEngine
import pandas as pd

print("=" * 80)
print("TESTING CATALYST CLASSIFICATION")
print("=" * 80)

# Create a minimal engine just to test the classifier
df = pd.DataFrame({'ticker': ['TEST'], 'sector': ['Healthcare'], 'market_cap': [1e9]})
engine = AIPortfolioEngine(df)

# Test cases
test_cases = [
    ("PDUFA", "REGULATORY", "Regulatory"),
    ("Phase2 Data", "CLINICAL", "Clinical"),
    ("Phase3 Readout", "CLINICAL", "Clinical"),
    ("sBLA", "REGULATORY", "Regulatory"),
    ("sNDA", "REGULATORY", "Regulatory"),
    ("Product Launch", "COMMERCIAL", "Commercial"),
    ("Topline Results", "CLINICAL", "Clinical"),
    ("FDA Approval", "REGULATORY", "Regulatory"),
    (None, "UNKNOWN", "Unknown Catalyst"),
    ("Something Random", "UNKNOWN", "Unknown Catalyst"),
]

print("\nTest Results:")
print("-" * 80)
print(f"{'Input':<30} {'Expected Class':<15} {'Got Class':<15} {'Expected Label':<20} {'Got Label':<20} {'Status':<10}")
print("-" * 80)

all_passed = True
for catalyst_type, expected_class, expected_label in test_cases:
    actual_class, actual_label = engine._classify_catalyst(catalyst_type)

    passed = (actual_class == expected_class and actual_label == expected_label)
    status = "✅ PASS" if passed else "❌ FAIL"

    if not passed:
        all_passed = False

    input_str = str(catalyst_type) if catalyst_type else "None"
    print(
        f"{input_str:<30} {expected_class:<15} {actual_class:<15} {expected_label:<20} {actual_label:<20} {status:<10}")

print("-" * 80)

if all_passed:
    print("\n✅ ALL TESTS PASSED - Catalyst classification working correctly!")
else:
    print("\n❌ SOME TESTS FAILED - Check the implementation")

print("\n" + "=" * 80)
print("NEXT: Test with real FDA calendar data")
print("=" * 80)

# Show what the real FDA events will be classified as
print("\nReal FDA Calendar Classifications:")
print("-" * 80)

real_events = [
    "PDUFA",
    "sNDA",
    "sBLA",
    "Phase3 Extension",
    "Phase2 Data",
    "Product Enhancement",
    "Product Launch",
    "Phase3 Readout",
]

for event in real_events:
    cat_class, cat_label = engine._classify_catalyst(event)
    print(f"{event:<30} → {cat_class:<15} ({cat_label})")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)