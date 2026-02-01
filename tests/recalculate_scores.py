"""
Recalculate Scores for Specific Tickers
========================================
This script recalculates Technical Score, Fundamental Score, and Total Score
for tickers that already have data but are missing these scores.

It does NOT:
- Fetch all 200+ stocks from watchlists
- Re-run sentiment analysis (keeps existing)
- Re-run options flow analysis (keeps existing)

It DOES:
- Run TechnicalAnalyzer to get Technical Score
- Recalculate Fundamental Score from existing DB data (Finviz data already loaded)
- Recalculate Total Score with proper weighting
- Update screener_scores table

Usage:
    python recalculate_scores.py                    # All 82 tickers
    python recalculate_scores.py NVAX CERT ADPT    # Specific tickers
    python recalculate_scores.py --dry-run         # Preview only
"""

import sys
import time
import pandas as pd
from datetime import date
from typing import Optional, Dict, Any, List

# Tickers that need score recalculation
TICKERS_TO_UPDATE = [
    "ABEO", "CRMD", "NVAX", "CTMX", "MDXG", "XOMA", "STOK", "OCGN", "ERAS", "DAWN",
    "GPCR", "LXRX", "ALT", "CMPX", "ORIC", "PGEN", "NRIX", "BCRX", "CRVS", "TRVI",
    "FTRE", "NEOG", "ARDX", "TERN", "XERS", "ESPR", "LRMR", "EWTX", "ABSI", "KURA",
    "TWST", "REPL", "RCKT", "ATAI", "VIR", "PROK", "ALDX", "KALV", "AMLX", "PSNL",
    "AQST", "MRVI", "ADPT", "JANX", "IMRX", "DYN", "MYGN", "SANA", "AUTL", "SDGR",
    "TXG", "OPK", "PCRX", "FULC", "SPRY", "VRDN", "CDTX", "VSTM", "WVE", "TNGX",
    "ABCL", "PACB", "GOSS", "NTLA", "SVRA", "AKBA", "COGT", "BBNX", "PRME", "QURE",
    "DVAX", "ELVN", "RLAY", "ATYR", "SRDX", "ATXS", "ABUS", "RZLT", "SRPT", "MNKD",
    "CERT", "AVDL"
]

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def to_native(val):
    """Convert numpy types to Python native and handle special values."""
    if val is None:
        return None
    if hasattr(val, 'item'):
        val = val.item()
    if isinstance(val, float):
        if pd.isna(val) or val == float('inf') or val == float('-inf'):
            return None
    return val


def get_technical_score(ticker: str) -> Optional[float]:
    """Run TechnicalAnalyzer to get technical score."""
    try:
        from src.analytics.technical_analysis import TechnicalAnalyzer
        ta = TechnicalAnalyzer()
        result = ta.analyze_ticker(ticker)
        if result:
            # Result is a TechnicalAnalysis object, not a dict
            score = getattr(result, 'technical_score', None)
            return score
    except ImportError:
        print(f"    {Colors.RED}TechnicalAnalyzer not available{Colors.END}")
    except Exception as e:
        print(f"    {Colors.YELLOW}Technical analysis failed: {e}{Colors.END}")
    return None


def get_fundamentals_from_db(ticker: str, today: date) -> Dict[str, Any]:
    """Get fundamental data from database (already populated by update_missing_data.py)."""
    try:
        from src.db.connection import get_connection

        with get_connection() as conn:
            with conn.cursor() as cur:
                # Get latest fundamentals
                cur.execute("""
                    SELECT pe_ratio, forward_pe, peg_ratio, roe, dividend_yield,
                           revenue_growth, earnings_growth, debt_to_equity, current_ratio
                    FROM fundamentals 
                    WHERE ticker = %s 
                    ORDER BY date DESC 
                    LIMIT 1
                """, (ticker,))
                row = cur.fetchone()

                if row:
                    return {
                        'pe_ratio': row[0],
                        'forward_pe': row[1],
                        'peg_ratio': row[2],
                        'roe': row[3],
                        'dividend_yield': row[4],
                        'revenue_growth': row[5],
                        'earnings_growth': row[6],
                        'debt_to_equity': row[7],
                        'current_ratio': row[8]
                    }
    except Exception as e:
        print(f"    {Colors.YELLOW}DB fundamentals fetch failed: {e}{Colors.END}")

    return {}


def get_existing_scores(ticker: str, today: date) -> Dict[str, Any]:
    """Get existing scores from screener_scores table."""
    try:
        from src.db.connection import get_connection

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT sentiment_score, options_flow_score, short_squeeze_score,
                           fundamental_score, technical_score, growth_score, dividend_score
                    FROM screener_scores 
                    WHERE ticker = %s AND date = %s
                """, (ticker, today))
                row = cur.fetchone()

                if row:
                    return {
                        'sentiment_score': row[0],
                        'options_flow_score': row[1],
                        'short_squeeze_score': row[2],
                        'fundamental_score': row[3],
                        'technical_score': row[4],
                        'growth_score': row[5],
                        'dividend_score': row[6]
                    }
    except Exception as e:
        print(f"    {Colors.YELLOW}DB scores fetch failed: {e}{Colors.END}")

    return {}


def calculate_fundamental_score(fund_data: Dict[str, Any]) -> Optional[float]:
    """Calculate fundamental score from data."""
    if not fund_data:
        return None

    pe = fund_data.get('pe_ratio')
    peg = fund_data.get('peg_ratio')
    roe = fund_data.get('roe')
    dividend_yield = fund_data.get('dividend_yield')
    revenue_growth = fund_data.get('revenue_growth')
    earnings_growth = fund_data.get('earnings_growth')
    debt_to_equity = fund_data.get('debt_to_equity')
    current_ratio = fund_data.get('current_ratio')

    fund_points = 50  # Start neutral

    # PE analysis
    if pe and pe > 0:
        if pe < 15:
            fund_points += 10
        elif pe < 25:
            fund_points += 5
        elif pe > 50:
            fund_points -= 10

    # PEG analysis
    if peg and peg > 0:
        if peg < 1:
            fund_points += 10
        elif peg < 2:
            fund_points += 5
        elif peg > 3:
            fund_points -= 5

    # Profitability
    if roe:
        if roe > 0.25:
            fund_points += 10
        elif roe > 0.15:
            fund_points += 5
        elif roe < 0:
            fund_points -= 5

    # Financial health
    if current_ratio:
        if current_ratio > 2:
            fund_points += 5
        elif current_ratio < 1:
            fund_points -= 5

    if debt_to_equity:
        if debt_to_equity < 0.5:
            fund_points += 5
        elif debt_to_equity > 2:
            fund_points -= 5

    return max(0, min(100, fund_points))


def calculate_growth_score(fund_data: Dict[str, Any]) -> Optional[float]:
    """Calculate growth score from data."""
    if not fund_data:
        return None

    revenue_growth = fund_data.get('revenue_growth')
    earnings_growth = fund_data.get('earnings_growth')

    growth_points = 50

    if revenue_growth:
        if revenue_growth > 0.2:
            growth_points += 20
        elif revenue_growth > 0.1:
            growth_points += 15
        elif revenue_growth > 0:
            growth_points += 5
        elif revenue_growth < -0.1:
            growth_points -= 15

    if earnings_growth:
        if earnings_growth > 0.2:
            growth_points += 20
        elif earnings_growth > 0.1:
            growth_points += 15
        elif earnings_growth > 0:
            growth_points += 5
        elif earnings_growth < -0.1:
            growth_points -= 15

    return max(0, min(100, growth_points))


def calculate_dividend_score(fund_data: Dict[str, Any]) -> Optional[float]:
    """Calculate dividend score from data."""
    dividend_yield = fund_data.get('dividend_yield')

    dividend_points = 50
    if dividend_yield:
        if dividend_yield > 0.04:
            dividend_points += 30
        elif dividend_yield > 0.02:
            dividend_points += 20
        elif dividend_yield > 0.01:
            dividend_points += 10

    return max(0, min(100, dividend_points))


def calculate_total_score(sentiment: float, fundamental: float, technical: float,
                          options: float, squeeze: float) -> float:
    """Calculate weighted total score."""
    available_scores = []
    available_weights = []

    if sentiment is not None:
        available_scores.append(sentiment * 0.25)
        available_weights.append(0.25)
    if fundamental is not None:
        available_scores.append(fundamental * 0.25)
        available_weights.append(0.25)
    if technical is not None:
        available_scores.append(technical * 0.25)
        available_weights.append(0.25)
    if options is not None:
        available_scores.append(options * 0.15)
        available_weights.append(0.15)
    if squeeze is not None:
        available_scores.append(squeeze * 0.10)
        available_weights.append(0.10)

    if available_weights:
        total_weight = sum(available_weights)
        total_score = round(sum(available_scores) / total_weight) if total_weight > 0 else 50
        return max(0, min(100, total_score))

    return 50


def update_scores_in_db(ticker: str, today: date, scores: Dict[str, Any], dry_run: bool = False) -> bool:
    """Update screener_scores table with recalculated scores."""
    if dry_run:
        print(f"    {Colors.CYAN}[DRY RUN] Would update: Tech={scores.get('technical')}, Fund={scores.get('fundamental')}, Total={scores.get('total')}{Colors.END}")
        return True

    try:
        from src.db.connection import get_connection

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE screener_scores SET
                        technical_score = COALESCE(%s, technical_score),
                        fundamental_score = COALESCE(%s, fundamental_score),
                        growth_score = COALESCE(%s, growth_score),
                        dividend_score = COALESCE(%s, dividend_score),
                        total_score = %s
                    WHERE ticker = %s AND date = %s
                """, (
                    to_native(scores.get('technical')),
                    to_native(scores.get('fundamental')),
                    to_native(scores.get('growth')),
                    to_native(scores.get('dividend')),
                    to_native(scores.get('total')),
                    ticker, today
                ))
            conn.commit()
        return True
    except Exception as e:
        print(f"    {Colors.RED}DB update failed: {e}{Colors.END}")
        return False


def process_ticker(ticker: str, dry_run: bool = False) -> Dict:
    """Process a single ticker."""
    print(f"\n{Colors.BOLD}[{ticker}]{Colors.END}")

    today = date.today()

    # Step 1: Get existing scores (sentiment, options, squeeze)
    existing = get_existing_scores(ticker, today)
    if not existing:
        print(f"  {Colors.YELLOW}No existing scores found for today{Colors.END}")
        return {'ticker': ticker, 'success': False, 'reason': 'No existing scores'}

    sentiment = existing.get('sentiment_score')
    options = existing.get('options_flow_score')
    squeeze = existing.get('short_squeeze_score')

    print(f"  Existing: Sent={sentiment}, Opts={options}, Squeeze={squeeze}")

    # Step 2: Get Technical Score (run fresh)
    print(f"  Running Technical Analysis...")
    technical = get_technical_score(ticker)
    if technical:
        print(f"  {Colors.GREEN}✓ Technical: {technical}{Colors.END}")
    else:
        print(f"  {Colors.YELLOW}○ Technical: None{Colors.END}")

    # Step 3: Get Fundamentals from DB and recalculate scores
    fund_data = get_fundamentals_from_db(ticker, today)
    if fund_data:
        fundamental = calculate_fundamental_score(fund_data)
        growth = calculate_growth_score(fund_data)
        dividend = calculate_dividend_score(fund_data)
        print(f"  {Colors.GREEN}✓ Fundamental: {fundamental}, Growth: {growth}, Dividend: {dividend}{Colors.END}")
    else:
        fundamental = existing.get('fundamental_score', 50)
        growth = existing.get('growth_score', 50)
        dividend = existing.get('dividend_score', 50)
        print(f"  {Colors.YELLOW}○ Using existing fundamental scores{Colors.END}")

    # Step 4: Calculate Total Score
    total = calculate_total_score(sentiment, fundamental, technical, options, squeeze)
    print(f"  {Colors.CYAN}Total Score: {total}{Colors.END}")

    # Step 5: Update database
    scores = {
        'technical': technical,
        'fundamental': fundamental,
        'growth': growth,
        'dividend': dividend,
        'total': total
    }

    if update_scores_in_db(ticker, today, scores, dry_run):
        print(f"  {Colors.GREEN}✓ Updated{Colors.END}")
        return {'ticker': ticker, 'success': True, 'scores': scores}
    else:
        return {'ticker': ticker, 'success': False, 'reason': 'DB error'}


def main():
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}Recalculate Scores - Technical + Fundamental{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")

    # Parse arguments
    tickers = []
    dry_run = False

    for arg in sys.argv[1:]:
        if arg.lower() == '--dry-run':
            dry_run = True
        elif len(arg) <= 6:
            tickers.append(arg.upper())

    if not tickers:
        tickers = TICKERS_TO_UPDATE
        print(f"\nRecalculating scores for all {len(tickers)} tickers")
    else:
        print(f"\nRecalculating scores for {len(tickers)} tickers: {', '.join(tickers)}")

    if dry_run:
        print(f"{Colors.YELLOW}DRY RUN MODE - No changes will be saved{Colors.END}")

    # Check TechnicalAnalyzer availability
    try:
        from src.analytics.technical_analysis import TechnicalAnalyzer
        print(f"{Colors.GREEN}✓ TechnicalAnalyzer available{Colors.END}")
    except ImportError:
        print(f"{Colors.RED}✗ TechnicalAnalyzer NOT available - Technical scores will be None{Colors.END}")

    # Process tickers
    results = []
    success_count = 0

    for i, ticker in enumerate(tickers):
        print(f"\n[{i+1}/{len(tickers)}]", end="")
        result = process_ticker(ticker, dry_run)
        results.append(result)
        if result['success']:
            success_count += 1

        # Small delay between tickers (for technical analysis API calls)
        if i < len(tickers) - 1:
            time.sleep(0.3)

    # Summary
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}SUMMARY{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"Total: {len(tickers)}")
    print(f"{Colors.GREEN}Success: {success_count}{Colors.END}")
    print(f"{Colors.RED}Failed: {len(tickers) - success_count}{Colors.END}")

    # Show score distribution
    if results:
        tech_scores = [r['scores']['technical'] for r in results if r.get('success') and r['scores'].get('technical')]
        fund_scores = [r['scores']['fundamental'] for r in results if r.get('success') and r['scores'].get('fundamental')]
        total_scores = [r['scores']['total'] for r in results if r.get('success') and r['scores'].get('total')]

        if tech_scores:
            print(f"\nTechnical Score: avg={sum(tech_scores)/len(tech_scores):.1f}, range={min(tech_scores)}-{max(tech_scores)}")
        if fund_scores:
            print(f"Fundamental Score: avg={sum(fund_scores)/len(fund_scores):.1f}, range={min(fund_scores)}-{max(fund_scores)}")
        if total_scores:
            print(f"Total Score: avg={sum(total_scores)/len(total_scores):.1f}, range={min(total_scores)}-{max(total_scores)}")

    if not dry_run and success_count > 0:
        print(f"\n{Colors.GREEN}✓ Scores updated! Reload signals table to see changes.{Colors.END}")


if __name__ == "__main__":
    main()