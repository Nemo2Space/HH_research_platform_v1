"""
Debug script for AI Portfolio Manager trade planning
Run from your project root: python debug_trade_planner.py
"""

import json
import os
from pathlib import Path


# Try to find your JSON portfolio file
def find_json_portfolios():
    """Find JSON portfolio files in common locations"""
    search_paths = [
        Path.home() / "portfolios",
        Path.cwd() / "portfolios",
        Path.cwd() / "data" / "portfolios",
        Path.cwd(),
    ]

    json_files = []
    for base in search_paths:
        if base.exists():
            json_files.extend(base.glob("**/*IBKR*.json"))
            json_files.extend(base.glob("**/portfolio*.json"))

    return list(set(json_files))[:5]  # Return up to 5 unique files


def load_portfolio_weights(json_path: str) -> dict:
    """Load weights from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    holdings = data if isinstance(data, list) else data.get('holdings', [])

    weights = {}
    for h in holdings:
        ticker = (h.get('symbol') or h.get('ticker') or '').upper().strip()
        # Check if weight is in percentage (0-100) or decimal (0-1)
        raw_weight = float(h.get('weight', 0))

        # Determine if it's percentage or decimal
        if raw_weight > 1.0:
            # It's a percentage, convert to decimal
            weight = raw_weight / 100.0
        else:
            weight = raw_weight

        if ticker and weight > 0:
            weights[ticker] = weight

    return weights


def analyze_weights(weights: dict):
    """Analyze weight distribution"""
    print("\n" + "=" * 60)
    print("WEIGHT ANALYSIS")
    print("=" * 60)

    total = sum(weights.values())
    print(f"\nTotal symbols: {len(weights)}")
    print(f"Sum of weights: {total:.4f} ({total * 100:.2f}%)")

    if total > 1.0:
        print(f"⚠️  OVER-ALLOCATED by {(total - 1) * 100:.2f}%")
        print("   Normalization will scale down all weights")
    elif total < 0.99:
        print(f"ℹ️  UNDER-ALLOCATED: {(1 - total) * 100:.2f}% will remain as cash")
    else:
        print("✅ Weights properly normalized")

    print(f"\nTop 10 weights:")
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]
    for sym, w in sorted_weights:
        print(f"  {sym:8s}: {w:.4f} ({w * 100:.2f}%)")

    return total


def simulate_normalization(weights: dict, total: float):
    """Simulate what trade_planner normalization does"""
    print("\n" + "=" * 60)
    print("NORMALIZATION SIMULATION")
    print("=" * 60)

    if total > 1.0:
        normalized = {k: v / total for k, v in weights.items()}
        new_total = sum(normalized.values())
        print(f"\nAfter normalization:")
        print(f"  Scale factor: {1 / total:.4f}")
        print(f"  New sum: {new_total:.4f} ({new_total * 100:.2f}%)")
        return normalized
    else:
        print("\nNo normalization needed (sum <= 100%)")
        return weights


def simulate_order_generation(weights: dict, nav: float, cash: float):
    """Simulate order generation"""
    print("\n" + "=" * 60)
    print(f"ORDER GENERATION SIMULATION (NAV=${nav:,.2f}, Cash=${cash:,.2f})")
    print("=" * 60)

    # Fetch some sample prices (you'd need real prices)
    sample_prices = {
        'MSFT': 420.0, 'GOOG': 180.0, 'NVDA': 140.0, 'AMZN': 220.0,
        'AMD': 120.0, 'TSM': 180.0, 'ASML': 750.0, 'AVGO': 180.0,
        'META': 600.0, 'BA': 180.0, 'AAPL': 230.0, 'GOOGL': 180.0,
    }

    total_buy_value = 0
    orders = []

    print(f"\nGenerating orders for {len(weights)} symbols...")

    for sym, target_w in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:15]:
        # Current weight is 0 (new portfolio)
        current_w = 0.0
        delta_w = target_w - current_w

        # Calculate order value
        delta_value = delta_w * nav

        # Get price (use sample or skip)
        price = sample_prices.get(sym, 100.0)  # Default $100 if unknown

        # Calculate shares
        qty = int(delta_value / price)

        if qty > 0:
            order_value = qty * price
            total_buy_value += order_value
            orders.append({
                'symbol': sym,
                'action': 'BUY',
                'qty': qty,
                'price': price,
                'value': order_value,
                'target_w': target_w,
            })
            print(f"  {sym:8s}: BUY {qty:4d} @ ${price:7.2f} = ${order_value:9.2f} (target: {target_w * 100:.2f}%)")

    print(f"\n{'=' * 40}")
    print(f"Total BUY value: ${total_buy_value:,.2f}")
    print(f"Available cash:  ${cash:,.2f}")

    if total_buy_value > cash:
        print(f"\n⚠️  BUY ORDERS EXCEED CASH!")
        print(f"   Overage: ${total_buy_value - cash:,.2f}")
        print(f"   This triggers the safety scaling in trade_planner.py")

        scale_factor = (cash * 0.995) / total_buy_value
        print(f"   Scale factor would be: {scale_factor:.4f}")

        if scale_factor < 0.1:
            print(f"\n❌ PROBLEM FOUND: Scale factor {scale_factor:.4f} is too small!")
            print("   This will make all orders nearly zero.")
    else:
        print(f"\n✅ Orders fit within available cash")
        print(f"   Remaining cash: ${cash - total_buy_value:,.2f}")

    return orders, total_buy_value


def main():
    print("=" * 60)
    print("AI PORTFOLIO MANAGER - TRADE PLANNER DEBUG")
    print("=" * 60)

    # Try to find JSON files
    json_files = find_json_portfolios()

    if json_files:
        print("\nFound portfolio files:")
        for i, f in enumerate(json_files):
            print(f"  [{i}] {f}")

        # Use first file or let user specify
        json_path = str(json_files[0])
        print(f"\nUsing: {json_path}")
    else:
        # Manual path entry
        json_path = input("\nEnter path to your portfolio JSON file: ").strip()

    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return

    # Load and analyze
    print(f"\nLoading: {json_path}")
    weights = load_portfolio_weights(json_path)

    if not weights:
        print("❌ No weights found in file!")
        return

    # Analyze
    total = analyze_weights(weights)

    # Normalize
    normalized_weights = simulate_normalization(weights, total)

    # Simulate order generation
    nav = 10000.0  # Your NAV
    cash = 10000.0  # Your cash (same as NAV for new portfolio)

    orders, total_buy = simulate_order_generation(normalized_weights, nav, cash)

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)

    if total > 1.0:
        print(f"\n1. Weights sum to {total * 100:.1f}% (should be 100%)")
        print("   → Normalization divides all weights by this sum")

    if total_buy > cash:
        print(f"\n2. After normalization, buy orders (${total_buy:,.0f}) > cash (${cash:,.0f})")
        print("   → Safety check scales down orders")
        print("   → If scale factor is too small, orders become 0")

    print("\n" + "=" * 60)
    print("RECOMMENDED FIX")
    print("=" * 60)
    print("""
The issue is that my safety check is too aggressive when:
- Weights sum to >100% AND
- Buy orders still exceed cash after normalization

The fix: Remove the aggressive safety scaling, rely only on normalization.
""")


if __name__ == "__main__":
    main()