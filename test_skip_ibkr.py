import sys, ast, time

for f in ["src/analytics/options_flow.py", "src/analytics/universe_scorer.py"]:
    try:
        ast.parse(open(f, encoding="utf-8").read())
        print(f"  SYNTAX OK: {f}")
    except SyntaxError as e:
        print(f"  SYNTAX ERROR in {f}: {e}")
        sys.exit(1)

print("\nChecking skip_ibkr parameter...")
import inspect
from src.analytics.options_flow import OptionsFlowAnalyzer
sig = inspect.signature(OptionsFlowAnalyzer.get_options_chain)
if "skip_ibkr" in sig.parameters:
    print("  OK: get_options_chain has skip_ibkr")
else:
    print("  MISSING! Params:", list(sig.parameters.keys()))
    sys.exit(1)

sig2 = inspect.signature(OptionsFlowAnalyzer.analyze_ticker)
if "skip_ibkr" in sig2.parameters:
    print("  OK: analyze_ticker has skip_ibkr")
else:
    print("  MISSING in analyze_ticker!")
    sys.exit(1)

from src.analytics.universe_scorer import UniverseScorer
sig3 = inspect.signature(UniverseScorer.__init__)
if "skip_ibkr" in sig3.parameters:
    print("  OK: UniverseScorer.__init__ has skip_ibkr")
else:
    print("  MISSING in UniverseScorer.__init__!")
    sys.exit(1)

print("\nTiming test with AAPL...")
analyzer = OptionsFlowAnalyzer()

start = time.time()
calls, puts, price, src = analyzer.get_options_chain("AAPL", skip_ibkr=False)
t_ibkr = time.time() - start
print(f"  With IBKR attempt: {t_ibkr:.1f}s, source={src}, price=${price:.2f}")

start = time.time()
calls, puts, price, src = analyzer.get_options_chain("AAPL", skip_ibkr=True)
t_skip = time.time() - start
print(f"  Skip IBKR:         {t_skip:.1f}s, source={src}, price=${price:.2f}")
print(f"  Time saved:        {t_ibkr - t_skip:.1f}s")

print("\nAll checks passed!")
