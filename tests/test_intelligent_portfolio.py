"""
Test Intelligent Portfolio Builder
===================================
Demonstrates the new strategy-aware portfolio construction.
"""

import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np

# Your biotech tickers
BIOTECH_TICKERS = [
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
print("INTELLIGENT PORTFOLIO BUILDER TEST")
print("=" * 80)

# Load universe directly from database
print("\n1. Loading stock universe from database...")

def get_db_connection():
    """Get database connection."""
    try:
        from src.db.connection import get_connection
        cm = get_connection()
        return cm.__enter__()
    except (ImportError, AttributeError):
        import psycopg2
        return psycopg2.connect(
            host="localhost", port=5432, dbname="alpha_platform",
            user="alpha", password="alpha_secure_2024"
        )

def load_universe():
    """Load stock universe directly from database."""
    conn = get_db_connection()

    # First, check what columns exist in fundamentals table
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'fundamentals'
        """)
        fund_cols = [r[0] for r in cur.fetchall()]

        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'screener_scores'
        """)
        score_cols = [r[0] for r in cur.fetchall()]

    print(f"   Found {len(fund_cols)} columns in fundamentals, {len(score_cols)} in screener_scores")

    # Build dynamic query based on available columns
    fund_columns = []
    desired_fund_cols = [
        'market_cap', 'pe_ratio', 'forward_pe', 'pb_ratio', 'ps_ratio', 'peg_ratio',
        'profit_margin', 'operating_margin', 'gross_margin', 'roe', 'roa',
        'revenue_growth', 'earnings_growth', 'eps_growth', 'dividend_yield',
        'dividend_payout_ratio', 'current_ratio', 'debt_to_equity', 'free_cash_flow',
        'eps', 'sector', 'industry', 'cash', 'total_assets', 'total_debt'
    ]

    for col in desired_fund_cols:
        if col in fund_cols:
            fund_columns.append(f'f.{col}')

    fund_select = ', '.join(fund_columns) if fund_columns else 'f.market_cap'

    query = f"""
    WITH latest_scores AS (
        SELECT DISTINCT ON (ticker)
            ticker, date as score_date, sentiment_score, sentiment_weighted,
            fundamental_score, growth_score, dividend_score, technical_score,
            gap_score, gap_type, likelihood_score, analyst_positivity,
            target_upside_pct, insider_signal, institutional_signal,
            composite_score, total_score, options_flow_score, short_squeeze_score,
            options_sentiment, squeeze_risk, earnings_adjustment, earnings_score,
            earnings_signal, days_to_earnings
        FROM screener_scores
        ORDER BY ticker, date DESC
    ),
    latest_fundamentals AS (
        SELECT DISTINCT ON (ticker)
            ticker, {', '.join([c for c in desired_fund_cols if c in fund_cols])}
        FROM fundamentals
        ORDER BY ticker, date DESC
    )
    SELECT 
        s.*, 
        {fund_select}
    FROM latest_scores s
    LEFT JOIN latest_fundamentals f ON s.ticker = f.ticker
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # Add company_name if missing
    if 'company_name' not in df.columns:
        df['company_name'] = df['ticker']

    return df

try:
    df = load_universe()
    print(f"   ✓ Loaded {len(df)} stocks")
except Exception as e:
    print(f"   ✗ Error loading universe: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Import the intelligent builder
print("\n2. Loading Intelligent Portfolio Builder...")
try:
    from portfolio_engine_v3 import (
        IntelligentPortfolioEngine,
        InvestmentStrategy,
        STRATEGY_SCORING_MODELS,
        PortfolioIntent
    )
    print("   ✓ Loaded from portfolio_engine_v3")
except ImportError:
    # Try from dashboard if installed
    from dashboard.portfolio_engine import (
        IntelligentPortfolioEngine,
        InvestmentStrategy,
        STRATEGY_SCORING_MODELS,
        PortfolioIntent
    )
    print("   ✓ Loaded from dashboard.portfolio_engine")

# Show available strategies
print("\n3. Available Investment Strategies:")
for strategy in InvestmentStrategy:
    model = STRATEGY_SCORING_MODELS.get(strategy)
    if model:
        print(f"   • {strategy.value}: {model['name']}")

# Test 1: Biotech Growth Portfolio
print("\n" + "=" * 80)
print("TEST 1: BIOTECH GROWTH PORTFOLIO")
print("=" * 80)

user_request = """
Build a growth portfolio for biotech companies focusing on near-term catalysts.
I want to maximize upside while managing binary risk.
Select 15-18 stocks from my biotech watchlist.
"""

print(f"\nUser Request:\n{user_request}")

# Create intent
intent = PortfolioIntent(
    objective="biotech_growth",
    risk_level="aggressive",
    max_holdings=18,
    min_holdings=12,
    max_position_pct=8,
    tickers_include=BIOTECH_TICKERS,
    restrict_to_tickers=False,
)

# Build portfolio
engine = IntelligentPortfolioEngine(df)
result = engine.build_portfolio(intent, user_request=user_request)

print(f"\n✓ Strategy Detected: {result.strategy_name}")
print(f"✓ Holdings: {result.total_positions if hasattr(result, 'total_positions') else result.num_holdings}")
print(f"✓ Invested: {result.invested_value/result.total_value*100:.1f}%")
print(f"✓ Cash Buffer: {result.cash_value/result.total_value*100:.1f}%")

print("\n--- TOP 10 HOLDINGS ---")
print(f"{'Ticker':<8} {'Weight':>7} {'Conv':>8} {'Score':>6} {'Market Cap':>12} {'Rationale'}")
print("-" * 70)
for h in sorted(result.holdings, key=lambda x: -x.weight_pct)[:10]:
    mc_str = f"${h.market_cap/1e9:.1f}B" if h.market_cap and h.market_cap >= 1e9 else (f"${h.market_cap/1e6:.0f}M" if h.market_cap else "N/A")
    conv = h.conviction or "N/A"
    rationale = (h.rationale or "")[:30]
    print(f"{h.ticker:<8} {h.weight_pct:>6.1f}% {conv:>8} {h.score:>6.0f} {mc_str:>12} {rationale}")

print("\n--- SECTOR ALLOCATION ---")
for sector, weight in sorted(result.sectors.items(), key=lambda x: -x[1]):
    print(f"   {sector}: {weight:.1f}%")

print("\n--- KEY PORTFOLIO RISKS ---")
if result.key_risks:
    for risk in result.key_risks:
        print(f"   • {risk}")

# Show detailed analysis for top holding
print("\n--- DETAILED ANALYSIS: TOP HOLDING ---")
top = sorted(result.holdings, key=lambda x: -x.weight_pct)[0]
print(f"\n{top.ticker} - {top.company_name}")
print(f"Weight: {top.weight_pct:.1f}% | Conviction: {top.conviction or 'N/A'} | Score: {top.score:.0f}")
if top.score_breakdown:
    print(f"\nScore Breakdown:")
    for factor, score in sorted(top.score_breakdown.items(), key=lambda x: -x[1])[:5]:
        print(f"   {factor}: {score:.1f}")
if top.bull_case:
    print(f"\nBull Case:")
    for point in top.bull_case[:3]:
        print(f"   ✓ {point}")
if top.bear_case:
    print(f"\nBear Case:")
    for point in top.bear_case[:3]:
        print(f"   ✗ {point}")
if top.catalysts:
    print(f"\nKey Catalysts:")
    for cat in top.catalysts[:3]:
        print(f"   → {cat}")

# Test 2: Different strategy - Value
print("\n" + "=" * 80)
print("TEST 2: VALUE PORTFOLIO (Full Universe)")
print("=" * 80)

value_request = """
Build a conservative value portfolio focusing on undervalued, 
fundamentally strong companies with dividend support.
"""

print(f"\nUser Request:\n{value_request}")

value_intent = PortfolioIntent(
    objective="value",
    risk_level="moderate",
    max_holdings=20,
)

value_engine = IntelligentPortfolioEngine(df)
value_result = value_engine.build_portfolio(value_intent, user_request=value_request)

print(f"\n✓ Strategy Detected: {value_result.strategy_name}")
print(f"✓ Holdings: {value_result.num_holdings}")

print("\n--- TOP 10 HOLDINGS ---")
for h in sorted(value_result.holdings, key=lambda x: -x.weight_pct)[:10]:
    mc_str = f"${h.market_cap/1e9:.1f}B" if h.market_cap and h.market_cap >= 1e9 else (f"${h.market_cap/1e6:.0f}M" if h.market_cap else "N/A")
    conv = h.conviction or "N/A"
    print(f"{h.ticker:<8} {h.weight_pct:>6.1f}% {conv:>8} {mc_str:>12}")

# Generate full markdown report
print("\n" + "=" * 80)
print("GENERATING FULL MARKDOWN REPORT")
print("=" * 80)

# Save report to file
report_md = result.to_markdown()
with open('portfolio_report.md', 'w', encoding='utf-8') as f:
    f.write(report_md)
print("\n✓ Saved full report to: portfolio_report.md")

# Print excerpt
print("\n--- REPORT EXCERPT ---")
print(report_md[:2500])
print("\n... [see portfolio_report.md for full report]")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)