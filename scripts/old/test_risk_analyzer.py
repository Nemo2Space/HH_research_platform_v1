"""
Test script for Risk Analyzer module.

Run this to verify the risk analyzer works with your IBKR data.

Usage:
    python scripts/test_risk_analyzer.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.risk_analyzer import RiskAnalyzer, analyze_portfolio_risk
from src.broker.ibkr_utils import load_ibkr_data_cached


def test_with_ibkr():
    """Test risk analyzer with real IBKR data."""
    print("=" * 60)
    print("RISK ANALYZER TEST")
    print("=" * 60)

    # 1. Load IBKR data
    print("\n1. Loading IBKR data...")

    # You may need to adjust the account_id
    ibkr_data = load_ibkr_data_cached(
        account_id="",  # Leave empty to use first account
        host="127.0.0.1",
        port=7496,
        fetch_live_prices=True
    )

    if ibkr_data.get('error'):
        print(f"   ❌ Error: {ibkr_data['error']}")
        print("   Make sure TWS/Gateway is running!")
        return

    positions = ibkr_data.get('positions', [])
    summary = ibkr_data.get('summary', {})

    print(f"   ✅ Loaded {len(positions)} positions")
    print(f"   Portfolio value: ${summary.get('net_liquidation', 0):,.2f}")

    # 2. Run risk analysis
    print("\n2. Running risk analysis...")

    analyzer = RiskAnalyzer()
    metrics = analyzer.analyze_portfolio(positions, summary)

    # 3. Display results
    print("\n" + "=" * 60)
    print("RISK METRICS RESULTS")
    print("=" * 60)

    print(f"\n📊 RISK SCORE: {metrics.risk_score} | RISK LEVEL: {metrics.risk_level}")
    print(f"   Total alerts: {metrics.total_alerts}")

    print(f"\n📈 CONCENTRATION:")
    print(f"   Top 5 concentration: {metrics.top_5_concentration}%")
    print(f"   Top 10 concentration: {metrics.top_10_concentration}%")
    print(f"   Largest position: {metrics.largest_position[0]} ({metrics.largest_position[1]}%)")

    print(f"\n🏢 SECTOR EXPOSURE:")
    for sector, weight in list(metrics.sector_breakdown.items())[:5]:
        marker = "⚠️" if weight > 30 else "  "
        print(f"   {marker} {sector}: {weight}%")

    print(f"\n📉 PORTFOLIO BETA: {metrics.portfolio_beta}")
    print(f"   {metrics.beta_interpretation}")

    if metrics.high_correlations:
        print(f"\n🔗 HIGH CORRELATIONS:")
        for sym1, sym2, corr in metrics.high_correlations[:5]:
            print(f"   {sym1} ↔ {sym2}: {corr:.0%}")

    # 4. Display alerts
    all_alerts = analyzer.get_all_alerts(metrics)

    if all_alerts:
        print(f"\n🚨 ALERTS ({len(all_alerts)} items):")
        print("-" * 50)
        for alert in all_alerts[:10]:  # Show first 10
            print(f"   {alert['icon']} [{alert['severity'].upper()}] {alert['message']}")
    else:
        print("\n✅ No alerts - portfolio looks healthy!")

    # 5. Signal conflicts detail
    if metrics.signal_conflicts:
        print(f"\n⚠️ SIGNAL CONFLICTS ({len(metrics.signal_conflicts)}):")
        print("-" * 50)
        for c in metrics.signal_conflicts:
            print(f"   {c['symbol']}: {c['signal']} signal for {c['days_with_signal']} days")
            print(f"      Weight: {c['weight']}%, P&L: {c['pnl_pct']}%")

    # 6. P&L candidates
    if metrics.profit_taking_candidates:
        print(f"\n💰 PROFIT TAKING CANDIDATES:")
        for p in metrics.profit_taking_candidates:
            print(f"   {p['symbol']}: +{p['pnl_pct']}% (${p['pnl_amount']:,.2f})")

    if metrics.stop_loss_candidates:
        print(f"\n⛔ REVIEW NEEDED (>20% loss):")
        for s in metrics.stop_loss_candidates:
            print(f"   {s['symbol']}: {s['pnl_pct']}% (${s['pnl_amount']:,.2f})")

    # 7. Generate AI summary
    print("\n" + "=" * 60)
    print("AI SUMMARY (for chat context)")
    print("=" * 60)
    summary_text = analyzer.generate_risk_summary(metrics)
    print(summary_text)

    print("\n✅ Test complete!")


def test_with_mock_data():
    """Test with mock data (no IBKR needed)."""
    print("=" * 60)
    print("RISK ANALYZER TEST (Mock Data)")
    print("=" * 60)

    # Mock positions
    mock_positions = [
        {'symbol': 'NVDA', 'quantity': 100, 'avg_cost': 400, 'current_price': 580,
         'market_value': 58000, 'unrealized_pnl': 18000, 'unrealized_pnl_pct': 45},
        {'symbol': 'AAPL', 'quantity': 200, 'avg_cost': 150, 'current_price': 175,
         'market_value': 35000, 'unrealized_pnl': 5000, 'unrealized_pnl_pct': 16.7},
        {'symbol': 'MSFT', 'quantity': 80, 'avg_cost': 350, 'current_price': 380,
         'market_value': 30400, 'unrealized_pnl': 2400, 'unrealized_pnl_pct': 8.6},
        {'symbol': 'GOOGL', 'quantity': 150, 'avg_cost': 140, 'current_price': 165,
         'market_value': 24750, 'unrealized_pnl': 3750, 'unrealized_pnl_pct': 17.9},
        {'symbol': 'AMZN', 'quantity': 100, 'avg_cost': 180, 'current_price': 195,
         'market_value': 19500, 'unrealized_pnl': 1500, 'unrealized_pnl_pct': 8.3},
        {'symbol': 'META', 'quantity': 50, 'avg_cost': 500, 'current_price': 550,
         'market_value': 27500, 'unrealized_pnl': 2500, 'unrealized_pnl_pct': 10},
        {'symbol': 'INTC', 'quantity': 500, 'avg_cost': 35, 'current_price': 25,
         'market_value': 12500, 'unrealized_pnl': -5000, 'unrealized_pnl_pct': -28.6},
        {'symbol': 'AMD', 'quantity': 100, 'avg_cost': 120, 'current_price': 140,
         'market_value': 14000, 'unrealized_pnl': 2000, 'unrealized_pnl_pct': 16.7},
    ]

    mock_summary = {
        'net_liquidation': 250000,
        'total_cash': 28350
    }

    print(f"\nMock portfolio: {len(mock_positions)} positions")
    print(f"Total value: ${mock_summary['net_liquidation']:,.2f}")

    # Run analysis
    analyzer = RiskAnalyzer()
    metrics = analyzer.analyze_portfolio(mock_positions, mock_summary)

    # Display results
    print(f"\n📊 RISK SCORE: {metrics.risk_score} | RISK LEVEL: {metrics.risk_level}")
    print(f"   Total alerts: {metrics.total_alerts}")

    print(f"\n📈 CONCENTRATION:")
    print(f"   Top 5: {metrics.top_5_concentration}%")
    print(f"   Largest: {metrics.largest_position[0]} ({metrics.largest_position[1]}%)")

    print(f"\n📉 PORTFOLIO BETA: {metrics.portfolio_beta}")

    # Alerts
    all_alerts = analyzer.get_all_alerts(metrics)
    if all_alerts:
        print(f"\n🚨 ALERTS:")
        for alert in all_alerts:
            print(f"   {alert['icon']} {alert['message']}")

    print("\n✅ Mock test complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test Risk Analyzer')
    parser.add_argument('--mock', action='store_true', help='Use mock data instead of IBKR')
    args = parser.parse_args()

    if args.mock:
        test_with_mock_data()
    else:
        test_with_ibkr()
