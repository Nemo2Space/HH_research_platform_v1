"""
Test AI Portfolio Engine V5 - Full Integration
===============================================
"""

import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np

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
print("AI PORTFOLIO ENGINE V5 - FULL INTEGRATION TEST")
print("=" * 80)

def get_db_connection():
    try:
        from src.db.connection import get_connection
        cm = get_connection()
        return cm.__enter__()
    except:
        import psycopg2
        return psycopg2.connect(
            host="localhost", port=5432, dbname="alpha_platform",
            user="alpha", password="alpha_secure_2024"
        )

print("\n1. Loading ALL AI data from database...")
conn = get_db_connection()

# Build comprehensive query with all AI tables
query = """
WITH latest_scores AS (
    SELECT DISTINCT ON (ticker)
        ticker, date as score_date,
        sentiment_score, fundamental_score, growth_score,
        technical_score, dividend_score, options_flow_score,
        short_squeeze_score, options_sentiment, squeeze_risk,
        days_to_earnings, target_upside_pct, analyst_positivity
    FROM screener_scores
    ORDER BY ticker, date DESC
),
latest_fundamentals AS (
    SELECT DISTINCT ON (ticker)
        ticker, market_cap, pe_ratio, pb_ratio,
        profit_margin, gross_margin, roe,
        revenue_growth, dividend_yield,
        current_ratio, debt_to_equity, free_cash_flow, sector
    FROM fundamentals
    ORDER BY ticker, date DESC
),
latest_ai AS (
    SELECT DISTINCT ON (ticker)
        ticker, ai_action, ai_confidence,
        bull_case as ai_bull_case, bear_case as ai_bear_case,
        key_risks as ai_key_risks, one_line_summary
    FROM ai_analysis
    ORDER BY ticker, analysis_date DESC
),
latest_ai_rec AS (
    SELECT DISTINCT ON (ticker)
        ticker, ai_probability
    FROM ai_recommendations
    ORDER BY ticker, created_at DESC
),
latest_committee AS (
    SELECT DISTINCT ON (ticker)
        ticker, verdict as committee_verdict,
        conviction as committee_conviction,
        rationale as committee_rationale
    FROM committee_decisions
    ORDER BY ticker, date DESC
),
latest_alpha AS (
    SELECT DISTINCT ON (ticker)
        ticker, predicted_probability as alpha_probability,
        alpha_signal as alpha_pred_signal
    FROM alpha_predictions
    ORDER BY ticker, prediction_date DESC
),
latest_enhanced AS (
    SELECT DISTINCT ON (ticker)
        ticker, insider_score, revision_score, earnings_surprise_score
    FROM enhanced_scores
    ORDER BY ticker, date DESC
),
latest_signals AS (
    SELECT DISTINCT ON (ticker)
        ticker, 
        signal_type as trading_signal_type,
        signal_strength as trading_signal_strength,
        signal_reason as trading_signal_reason
    FROM trading_signals
    ORDER BY ticker, date DESC
),
latest_earnings AS (
    SELECT DISTINCT ON (ticker)
        ticker, earnings_date as next_earnings_date,
        eps_estimate, guidance_direction
    FROM earnings_calendar
    WHERE earnings_date >= CURRENT_DATE
    ORDER BY ticker, earnings_date ASC
),
latest_fda AS (
    SELECT DISTINCT ON (ticker)
        ticker,
        expected_date as fda_expected_date,
        drug_name as fda_drug_name,
        catalyst_type as fda_catalyst_type,
        indication as fda_indication,
        priority as fda_priority,
        date_confirmed as fda_date_confirmed
    FROM fda_calendar
    WHERE expected_date >= CURRENT_DATE
    ORDER BY ticker, expected_date ASC
),
agent_fundamental AS (
    SELECT DISTINCT ON (ticker)
        ticker, buy_prob as agent_fundamental_buy_prob,
        confidence as agent_fundamental_confidence,
        rationale as agent_fundamental_rationale
    FROM agent_votes WHERE agent_role = 'fundamental'
    ORDER BY ticker, date DESC
),
agent_sentiment AS (
    SELECT DISTINCT ON (ticker)
        ticker, buy_prob as agent_sentiment_buy_prob,
        confidence as agent_sentiment_confidence,
        rationale as agent_sentiment_rationale
    FROM agent_votes WHERE agent_role = 'sentiment'
    ORDER BY ticker, date DESC
),
agent_technical AS (
    SELECT DISTINCT ON (ticker)
        ticker, buy_prob as agent_technical_buy_prob,
        confidence as agent_technical_confidence,
        rationale as agent_technical_rationale
    FROM agent_votes WHERE agent_role = 'technical'
    ORDER BY ticker, date DESC
),
agent_valuation AS (
    SELECT DISTINCT ON (ticker)
        ticker, buy_prob as agent_valuation_buy_prob,
        confidence as agent_valuation_confidence,
        rationale as agent_valuation_rationale
    FROM agent_votes WHERE agent_role = 'valuation'
    ORDER BY ticker, date DESC
)
SELECT 
    s.*,
    f.market_cap, f.pe_ratio, f.pb_ratio, f.profit_margin, f.gross_margin,
    f.roe, f.revenue_growth, f.dividend_yield, f.current_ratio,
    f.debt_to_equity, f.free_cash_flow, f.sector,
    ai.ai_action, ai.ai_confidence, ai.ai_bull_case, ai.ai_bear_case,
    ai.ai_key_risks, ai.one_line_summary,
    ar.ai_probability,
    c.committee_verdict, c.committee_conviction, c.committee_rationale,
    ap.alpha_probability, ap.alpha_pred_signal,
    e.insider_score, e.revision_score, e.earnings_surprise_score,
    sig.trading_signal_type, sig.trading_signal_strength, sig.trading_signal_reason,
    earn.next_earnings_date, earn.eps_estimate, earn.guidance_direction,
    fda.fda_expected_date, fda.fda_drug_name, fda.fda_catalyst_type, 
    fda.fda_indication, fda.fda_priority, fda.fda_date_confirmed,
    af.agent_fundamental_buy_prob, af.agent_fundamental_confidence, af.agent_fundamental_rationale,
    asent.agent_sentiment_buy_prob, asent.agent_sentiment_confidence, asent.agent_sentiment_rationale,
    at.agent_technical_buy_prob, at.agent_technical_confidence, at.agent_technical_rationale,
    av.agent_valuation_buy_prob, av.agent_valuation_confidence, av.agent_valuation_rationale
FROM latest_scores s
LEFT JOIN latest_fundamentals f ON s.ticker = f.ticker
LEFT JOIN latest_ai ai ON s.ticker = ai.ticker
LEFT JOIN latest_ai_rec ar ON s.ticker = ar.ticker
LEFT JOIN latest_committee c ON s.ticker = c.ticker
LEFT JOIN latest_alpha ap ON s.ticker = ap.ticker
LEFT JOIN latest_enhanced e ON s.ticker = e.ticker
LEFT JOIN latest_signals sig ON s.ticker = sig.ticker
LEFT JOIN latest_earnings earn ON s.ticker = earn.ticker
LEFT JOIN latest_fda fda ON s.ticker = fda.ticker
LEFT JOIN agent_fundamental af ON s.ticker = af.ticker
LEFT JOIN agent_sentiment asent ON s.ticker = asent.ticker
LEFT JOIN agent_technical at ON s.ticker = at.ticker
LEFT JOIN agent_valuation av ON s.ticker = av.ticker
"""

df = pd.read_sql(query, conn)
conn.close()

print(f"   ✓ Loaded {len(df)} stocks with {len(df.columns)} columns")

# Show data coverage
print("\n2. AI Data Coverage:")
ai_cols = {
    'ai_action': 'AI Analysis',
    'ai_probability': 'AI Probability',
    'committee_verdict': 'Committee Verdict',
    'committee_conviction': 'Committee Conviction',
    'alpha_probability': 'Alpha Predictions',
    'insider_score': 'Enhanced Scores',
    'trading_signal_type': 'Trading Signals',
    'agent_fundamental_buy_prob': 'Agent Votes (Fund)',
    'agent_sentiment_buy_prob': 'Agent Votes (Sent)',
    'next_earnings_date': 'Earnings Calendar',
    'fda_expected_date': 'FDA Calendar',
}

for col, name in ai_cols.items():
    if col in df.columns:
        count = df[col].notna().sum()
        pct = count / len(df) * 100
        print(f"   • {name}: {count}/{len(df)} ({pct:.0f}%)")

# Load engine
print("\n3. Loading AI Portfolio Engine V5...")
try:
    from portfolio_engine_v5 import AIPortfolioEngine, PortfolioIntent, InvestmentStrategy, STRATEGY_SCORING_MODELS
    print("   ✓ Loaded from portfolio_engine_v5")
except ImportError:
    from dashboard.portfolio_engine import AIPortfolioEngine, PortfolioIntent, InvestmentStrategy, STRATEGY_SCORING_MODELS
    print("   ✓ Loaded from dashboard.portfolio_engine")

print("\n4. Available Strategies:")
for s in InvestmentStrategy:
    m = STRATEGY_SCORING_MODELS.get(s)
    if m:
        print(f"   • {s.value}: {m['name']}")

# Test
print("\n" + "=" * 80)
print("TEST: BIOTECH GROWTH WITH FULL AI")
print("=" * 80)

intent = PortfolioIntent(
    objective="biotech_growth",
    max_holdings=15,
    tickers_include=BIOTECH_TICKERS,
    restrict_to_tickers=False,
)

engine = AIPortfolioEngine(df)
result = engine.build_portfolio(intent, "biotech growth portfolio with AI and FDA catalysts")

print(f"\n✓ Strategy: {result.strategy_name}")
print(f"✓ Holdings: {result.num_holdings}")
print(f"✓ Avg Score: {result.avg_score:.1f}")
print(f"✓ Avg AI Prob: {result.avg_ai_probability:.1f}%" if result.avg_ai_probability else "✓ Avg AI Prob: N/A")
print(f"✓ Avg Committee Conv: {result.avg_committee_conviction:.1f}" if result.avg_committee_conviction else "✓ Avg Committee Conv: N/A")
print(f"✓ AI Coverage: {result.ai_coverage_pct:.0f}%")
print(f"✓ Catalyst Coverage: {result.catalyst_coverage_pct:.0f}%")

print("\n--- HOLDINGS ---")
print(f"{'#':<3} {'Ticker':<8} {'Wt':>6} {'Score':>5} {'AI%':>5} {'Committee':>10} {'Signal':>8} {'Earnings':>8} {'FDA':>12} {'Conv':>6}")
print("-" * 95)
for i, h in enumerate(sorted(result.holdings, key=lambda x: -x.weight_pct), 1):
    ai_p = f"{h.ai_decision.ai_probability:.0f}" if h.ai_decision.ai_probability else "N/A"
    comm = h.ai_decision.committee_verdict or "N/A"
    sig = h.ai_decision.signal_type or "N/A"
    earn = f"{h.catalyst_info.days_to_earnings}d" if h.catalyst_info.days_to_earnings else "-"
    fda = f"{h.catalyst_info.days_to_fda}d" if h.catalyst_info.days_to_fda else "-"
    if h.catalyst_info.fda_catalyst_type:
        fda = f"{h.catalyst_info.fda_catalyst_type[:6]}:{fda}"
    print(f"{i:<3} {h.ticker:<8} {h.weight_pct:>5.1f}% {h.composite_score:>5.0f} {ai_p:>5} {comm:>10} {sig:>8} {earn:>8} {fda:>12} {h.conviction:>6}")

# Detail on top holding
print("\n--- TOP HOLDING DETAIL ---")
top = sorted(result.holdings, key=lambda x: -x.weight_pct)[0]
print(f"\n{top.ticker} | Score: {top.composite_score:.0f} | {top.conviction}")

ai = top.ai_decision
print(f"\nAI Decision Sources: {', '.join(ai.data_sources)}")
print(f"  AI Action: {ai.ai_action} ({ai.ai_confidence})")
print(f"  AI Probability: {ai.ai_probability:.1f}%" if ai.ai_probability else "  AI Probability: N/A")
print(f"  Committee: {ai.committee_verdict} (conviction: {ai.committee_conviction})" if ai.committee_verdict else "  Committee: N/A")
print(f"  Alpha Prob: {ai.alpha_probability:.1f}%" if ai.alpha_probability else "  Alpha Prob: N/A")

if ai.agent_votes:
    print(f"\n  Agent Votes:")
    for v in ai.agent_votes:
        print(f"    • {v.agent_role}: {v.buy_prob*100:.0f}% - {v.rationale[:50]}...")

if ai.insider_score is not None:
    print(f"\n  Enhanced: insider={ai.insider_score}, revision={ai.revision_score}, surprise={ai.earnings_surprise_score}")

print(f"\nScore Breakdown:")
for sd in sorted(top.score_details, key=lambda x: -x.weighted_contribution)[:8]:
    s = "✓" if sd.data_available else "✗"
    print(f"  {s} {sd.name:<22} {sd.value:>6.1f} x {sd.weight:>4.0%} = {sd.weighted_contribution:>5.1f}  [{sd.data_source}]")

if top.scores_missing:
    print(f"\n  Missing: {', '.join(top.scores_missing)}")

# Catalyst info
cat = top.catalyst_info
print(f"\nCatalysts:")
if cat.days_to_earnings:
    eps = f" (EPS est: ${cat.eps_estimate:.2f})" if cat.eps_estimate else ""
    print(f"  • Earnings: {cat.days_to_earnings} days{eps}")
if cat.days_to_fda:
    print(f"  • FDA ({cat.fda_catalyst_type}): {cat.days_to_fda} days - {cat.fda_drug} [{cat.fda_priority}]")
    if cat.catalyst_description:
        print(f"    {cat.catalyst_description}")
if not cat.days_to_earnings and not cat.days_to_fda:
    print(f"  • No near-term catalysts")

print(f"\nBull: {'; '.join(top.bull_case[:3])}")
print(f"Bear: {'; '.join(top.bear_case[:3])}")

# Save report
print("\n" + "=" * 80)
print("GENERATING REPORT")
print("=" * 80)

report = result.to_markdown()
with open('portfolio_report_v5.md', 'w', encoding='utf-8') as f:
    f.write(report)
print(f"✓ Saved: portfolio_report_v5.md ({len(report):,} chars)")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)