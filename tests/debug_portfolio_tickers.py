"""
Debug Portfolio ETF-Style Weighting
====================================
Test the new market-cap dominant weighting before integration.
"""

import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np

# Your biotech tickers
WHITELIST = [
    "ADPT", "DYN", "CERT", "COGT", "SRPT", "MNKD", "CDTX", "BCRX", "ARDX", "TXG",
    "VRDN", "TWST", "EWTX", "AVDL", "JANX", "SDGR", "ABCL", "NVAX", "STOK", "AMLX",
    "NTLA", "PCRX", "XERS", "GPCR", "NEOG", "ELVN", "BBNX", "PGEN", "QURE", "WVE",
    "OPK", "DVAX", "ORIC", "MDXG", "TRVI", "ATAI", "SPRY", "CRMD", "FTRE", "ABUS",
    "SANA", "TSHA", "PHAT", "IOVA", "GERN", "AVXL", "IMNM", "GOSS", "AKBA", "SVRA",
    "PROK", "DAWN", "TNGX", "KURA", "KALV", "VIR", "NRIX", "RLAY", "MRVI", "MYGN",
    "RZLT", "TERN", "CMPX", "AQST", "VSTM", "ATYR", "ESPR", "PRME", "PSNL", "SRDX",
    "XOMA", "REPL", "ERAS", "CRVS", "ATXS", "LXRX", "ALT", "ALDX", "ABEO", "CTMX",
    "OCGN", "LRMR", "RCKT", "PACB", "IMRX", "AUTL", "FULC", "ABSI"
]

print("=" * 80)
print("DEBUG: ETF-Style Portfolio Weighting Test")
print("=" * 80)

# Step 1: Load universe
print("\n1. Loading universe...")
from dashboard.portfolio_builder import get_latest_stock_universe
df = get_latest_stock_universe(force_refresh=True)
print(f"   Total stocks: {len(df)}")

# Filter to whitelist
whitelist_upper = [t.upper() for t in WHITELIST]
df_biotech = df[df['ticker'].str.upper().isin(whitelist_upper)].copy()
print(f"   Biotech whitelist stocks found: {len(df_biotech)}")

# Step 2: Check market_cap data
print("\n2. Market Cap Analysis:")
if 'market_cap' in df_biotech.columns:
    mc = df_biotech['market_cap'].dropna()
    print(f"   Stocks with market_cap: {len(mc)}/{len(df_biotech)}")
    print(f"   Range: ${mc.min()/1e6:.0f}M - ${mc.max()/1e9:.2f}B")
    print(f"   Median: ${mc.median()/1e6:.0f}M")

    # Show top 10 by market cap
    print("\n   Top 10 by Market Cap:")
    top_mc = df_biotech.nlargest(10, 'market_cap')[['ticker', 'market_cap', 'fundamental_score', 'sentiment_score']]
    for _, row in top_mc.iterrows():
        mc_str = f"${row['market_cap']/1e9:.2f}B" if row['market_cap'] >= 1e9 else f"${row['market_cap']/1e6:.0f}M"
        print(f"     {row['ticker']:6} | {mc_str:>10} | Fund: {row['fundamental_score']:.0f} | Sent: {row['sentiment_score']:.0f}")
else:
    print("   ✗ No market_cap column!")

# Step 3: Show new objective weights
print("\n3. New ETF-Style Objective Weights:")
from dashboard.portfolio_engine import OBJECTIVE_WEIGHTS, PortfolioObjective
for obj in [PortfolioObjective.BALANCED, PortfolioObjective.GROWTH, PortfolioObjective.VALUE]:
    weights = OBJECTIVE_WEIGHTS[obj]
    print(f"\n   {obj.value.upper()}:")
    for k, v in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"     {k:20} : {v:.0%}")

# Step 4: Run the portfolio engine
print("\n4. Running PortfolioEngine with ETF-style weights...")
from dashboard.portfolio_engine import PortfolioIntent, PortfolioEngine

intent = PortfolioIntent(
    objective="balanced",
    risk_level="aggressive",
    max_holdings=18,
    min_holdings=12,
    max_position_pct=10,
    max_sector_pct=35,
    tickers_include=WHITELIST,
    restrict_to_tickers=False,
    equal_weight=False  # Score-based weighting
)

print(f"   Intent: objective={intent.objective}, max_holdings={intent.max_holdings}")
print(f"   restrict_to_tickers={intent.restrict_to_tickers}")

try:
    engine = PortfolioEngine(df)
    result = engine.build_portfolio(intent)

    print(f"\n   SUCCESS: {result.success}")
    print(f"   Holdings: {result.num_holdings}")
    print(f"   Errors: {result.errors}")
    print(f"   Warnings: {result.warnings[:3] if result.warnings else 'None'}")

    if result.holdings:
        print(f"\n5. Portfolio Holdings (sorted by weight):")
        print(f"   {'Ticker':<8} {'Weight':>8} {'Score':>8} {'Market Cap':>12} {'Fund':>6} {'Sent':>6}")
        print("   " + "-" * 60)

        # Get market cap for each holding
        for h in sorted(result.holdings, key=lambda x: -x.weight_pct):
            mc = df_biotech[df_biotech['ticker'] == h.ticker]['market_cap'].values
            mc_str = f"${mc[0]/1e6:.0f}M" if len(mc) > 0 and mc[0] else "N/A"

            fund = df_biotech[df_biotech['ticker'] == h.ticker]['fundamental_score'].values
            fund_str = f"{fund[0]:.0f}" if len(fund) > 0 and pd.notna(fund[0]) else "N/A"

            sent = df_biotech[df_biotech['ticker'] == h.ticker]['sentiment_score'].values
            sent_str = f"{sent[0]:.0f}" if len(sent) > 0 and pd.notna(sent[0]) else "N/A"

            print(f"   {h.ticker:<8} {h.weight_pct:>7.2f}% {h.score:>8.1f} {mc_str:>12} {fund_str:>6} {sent_str:>6}")

        # Verify market cap correlation
        print("\n6. Market Cap Correlation Check:")
        tickers_selected = [h.ticker for h in result.holdings]
        selected_mc = df_biotech[df_biotech['ticker'].isin(tickers_selected)]['market_cap']
        not_selected_mc = df_biotech[~df_biotech['ticker'].isin(tickers_selected)]['market_cap']

        if len(selected_mc) > 0 and len(not_selected_mc) > 0:
            print(f"   Selected avg market cap: ${selected_mc.mean()/1e6:.0f}M")
            print(f"   Not selected avg market cap: ${not_selected_mc.mean()/1e6:.0f}M")

            if selected_mc.mean() > not_selected_mc.mean():
                print("   ✓ CORRECT: Selected stocks have higher avg market cap (ETF-style)")
            else:
                print("   ⚠ WARNING: Selected stocks have LOWER avg market cap than excluded")

except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Step 7: Compare with old total_score approach
print("\n7. Comparison: ETF-style vs Total Score ranking:")
print("\n   Top 10 by MARKET CAP (should be selected in ETF-style):")
top_mc = df_biotech.nlargest(10, 'market_cap')[['ticker', 'market_cap', 'total_score']]
for _, row in top_mc.iterrows():
    mc_str = f"${row['market_cap']/1e6:.0f}M"
    in_portfolio = "✓" if row['ticker'] in [h.ticker for h in result.holdings] else "✗"
    print(f"   {in_portfolio} {row['ticker']:6} | MC: {mc_str:>10} | Total: {row['total_score']:.0f}")

print("\n   Top 10 by TOTAL SCORE (old approach):")
top_ts = df_biotech.nlargest(10, 'total_score')[['ticker', 'market_cap', 'total_score']]
for _, row in top_ts.iterrows():
    mc_str = f"${row['market_cap']/1e6:.0f}M" if pd.notna(row['market_cap']) else "N/A"
    in_portfolio = "✓" if row['ticker'] in [h.ticker for h in result.holdings] else "✗"
    print(f"   {in_portfolio} {row['ticker']:6} | MC: {mc_str:>10} | Total: {row['total_score']:.0f}")

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)