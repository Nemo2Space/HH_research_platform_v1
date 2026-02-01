"""Test Institutional Signal Calculation"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.data.finviz import FinvizDataFetcher

f = FinvizDataFetcher()

tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'META', 'GOOGL', 'JPM', 'WMT', 'XOM', 'ZTS']

print(f"{'Ticker':<8} {'Inst%':>8} {'Insider%':>10} {'Short%':>8} {'Signal':>8}")
print("-" * 50)

for t in tickers:
    data = f.get_institutional_signal(t)
    inst = data.get('inst_own_pct', 0) or 0
    insider = data.get('insider_own_pct', 0) or 0
    short = data.get('short_float_pct', 0) or 0
    signal = data['institutional_signal']

    print(f"{t:<8} {inst:>8.1f} {insider:>10.1f} {short:>8.1f} {signal:>8}")

print("\n" + "=" * 50)
print("Signal Logic:")
print("  Base: 50")
print("  Inst Own >= 80%: +15")
print("  Inst Own >= 60%: +10")
print("  Inst Own >= 40%: +5")
print("  Inst Own < 20%: -10")
print("  Insider 5-30%: +10")
print("  Insider > 50%: -5")
print("  Short >= 20%: -15")
print("  Short >= 10%: -10")
print("  Short >= 5%: -5")