#!/usr/bin/env python3
"""
Test Script for Signal Hub Phase 1: Foundation

Tests:
1. UnifiedSignal model
2. SignalEngine initialization
3. Signal generation for stock
4. Signal generation for bond ETF
5. Market overview

Run: python scripts/test_signal_hub_phase1.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, date


def test_unified_signal_model():
    """Test the UnifiedSignal dataclass."""
    print("\n" + "=" * 60)
    print("TEST 1: UnifiedSignal Model")
    print("=" * 60)

    from src.core.unified_signal import (
        UnifiedSignal,
        SignalStrength,
        RiskLevel,
        AssetType,
    )

    # Create a signal
    signal = UnifiedSignal(
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        sector="Technology",
        asset_type=AssetType.STOCK,
        today_signal=SignalStrength.BUY,
        today_score=75,
        longterm_score=82,
        risk_level=RiskLevel.MEDIUM,
        technical_score=78,
        fundamental_score=85,
        sentiment_score=72,
        options_score=80,
        earnings_score=65,
        signal_reason="Strong technicals + bullish options flow",
    )

    print(f"✅ Created signal for {signal.ticker}")
    print(f"   Today: {signal.get_signal_emoji()} {signal.today_score}%")
    print(f"   Long-term: {signal.get_stars()} {signal.longterm_score}")
    print(f"   Risk: {signal.get_risk_emoji()} {signal.risk_level.value}")
    print(f"   Reason: {signal.signal_reason}")

    # Test serialization
    signal_dict = signal.to_dict()
    print(f"✅ Serialized to dict: {len(signal_dict)} fields")

    signal_json = signal.to_json()
    print(f"✅ Serialized to JSON: {len(signal_json)} chars")

    # Test deserialization
    restored = UnifiedSignal.from_dict(signal_dict)
    print(f"✅ Restored from dict: {restored.ticker} - {restored.today_signal.value}")

    # Test asset type detection
    print(f"\n   Asset type detection:")
    print(f"   NVDA → {UnifiedSignal.detect_asset_type('NVDA').value}")
    print(f"   TLT → {UnifiedSignal.detect_asset_type('TLT').value}")
    print(f"   ZROZ → {UnifiedSignal.detect_asset_type('ZROZ').value}")

    return True


def test_signal_engine_init():
    """Test SignalEngine initialization."""
    print("\n" + "=" * 60)
    print("TEST 2: SignalEngine Initialization")
    print("=" * 60)

    from src.core.signal_engine import SignalEngine

    engine = SignalEngine()

    components = engine._get_available_components()
    print(f"✅ SignalEngine initialized")
    print(f"   Available components: {components}")

    return True


def test_stock_signal_generation():
    """Test signal generation for a stock."""
    print("\n" + "=" * 60)
    print("TEST 3: Stock Signal Generation (AAPL)")
    print("=" * 60)

    from src.core.signal_engine import generate_signal

    signal = generate_signal("AAPL")

    print(f"✅ Generated signal for {signal.ticker}")
    print(f"   Company: {signal.company_name}")
    print(f"   Sector: {signal.sector}")
    print(f"   Price: ${signal.current_price:.2f}")
    print(f"")
    print(f"   Today Signal: {signal.get_signal_emoji()} {signal.today_score}%")
    print(f"   Long-term: {signal.get_stars()} {signal.longterm_score}")
    print(f"   Risk: {signal.get_risk_emoji()} {signal.risk_level.value}")
    print(f"")
    print(f"   Components:")
    print(f"     Technical:   {signal.technical_score} ({signal.technical_signal})")
    print(f"     Fundamental: {signal.fundamental_score} ({signal.fundamental_signal})")
    print(f"     Sentiment:   {signal.sentiment_score} ({signal.sentiment_signal})")
    print(f"     Options:     {signal.options_score} ({signal.options_signal})")
    print(f"     Earnings:    {signal.earnings_score} ({signal.earnings_signal})")
    print(f"")
    print(f"   Reason: {signal.signal_reason}")
    print(f"   Data Quality: {signal.data_quality}")
    print(f"   Components Used: {signal.components_available}")

    if signal.next_catalyst:
        print(f"   Next Catalyst: {signal.next_catalyst}")

    if signal.flags:
        print(f"   Flags: {', '.join(signal.flags)}")

    return True


def test_bond_signal_generation():
    """Test signal generation for a bond ETF."""
    print("\n" + "=" * 60)
    print("TEST 4: Bond ETF Signal Generation (TLT)")
    print("=" * 60)

    from src.core.signal_engine import generate_signal

    signal = generate_signal("TLT")

    print(f"✅ Generated signal for {signal.ticker}")
    print(f"   Asset Type: {signal.asset_type.value}")
    print(f"   Price: ${signal.current_price:.2f}")
    print(f"")
    print(f"   Today Signal: {signal.get_signal_emoji()} {signal.today_score}%")
    print(f"   Bond Score: {signal.bond_score} ({signal.bond_signal})")
    print(f"   Risk: {signal.get_risk_emoji()} {signal.risk_level.value}")
    print(f"")
    print(f"   Reason: {signal.signal_reason}")

    if signal.flags:
        print(f"   Flags: {', '.join(signal.flags)}")

    return True


def test_batch_signal_generation():
    """Test batch signal generation."""
    print("\n" + "=" * 60)
    print("TEST 5: Batch Signal Generation")
    print("=" * 60)

    from src.core.signal_engine import generate_signals

    tickers = ["NVDA", "MSFT", "TLT", "GOOGL"]

    print(f"Generating signals for: {tickers}")
    signals = generate_signals(tickers)

    print(f"✅ Generated {len(signals)} signals")
    print(f"")
    print(f"   {'Ticker':<8} {'Signal':<12} {'Today':<8} {'Long-term':<10} {'Risk':<10}")
    print(f"   {'-' * 8} {'-' * 12} {'-' * 8} {'-' * 10} {'-' * 10}")

    for ticker, signal in signals.items():
        print(
            f"   {ticker:<8} {signal.today_signal.value:<12} {signal.today_score:<8} {signal.longterm_score:<10} {signal.risk_level.value:<10}")

    return True


def test_market_overview():
    """Test market overview generation."""
    print("\n" + "=" * 60)
    print("TEST 6: Market Overview")
    print("=" * 60)

    from src.core.signal_engine import get_market_overview

    overview = get_market_overview()

    print(f"✅ Generated market overview")
    print(f"   Regime: {overview.regime} (Score: {overview.regime_score})")
    print(f"   VIX: {overview.vix}")
    print(f"   SPY Change: {overview.spy_change:+.2f}%")
    print(f"   High Impact Today: {overview.has_high_impact_today}")
    print(f"   Days to Fed: {overview.days_to_fed}")
    print(f"")
    print(f"   AI Summary: {overview.ai_summary}")

    if overview.economic_events_today:
        print(f"")
        print(f"   Today's Events:")
        for event in overview.economic_events_today[:3]:
            print(f"     • {event.get('name', 'Unknown')}")

    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SIGNAL HUB - PHASE 1 FOUNDATION TESTS")
    print("=" * 60)

    tests = [
        ("UnifiedSignal Model", test_unified_signal_model),
        ("SignalEngine Init", test_signal_engine_init),
        ("Stock Signal Generation", test_stock_signal_generation),
        ("Bond Signal Generation", test_bond_signal_generation),
        ("Batch Signal Generation", test_batch_signal_generation),
        ("Market Overview", test_market_overview),
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✅ PASS" if success else "❌ FAIL"))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"❌ ERROR: {str(e)[:50]}"))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, result in results:
        print(f"   {name:<30} {result}")

    passed = sum(1 for _, r in results if "PASS" in r)
    print(f"\n   Total: {passed}/{len(results)} tests passed")

    return passed == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)