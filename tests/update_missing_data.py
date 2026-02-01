"""
Update Missing Data for Specific Tickers
=========================================
Directly updates the database with fundamentals, analyst ratings,
price targets from yfinance + Finviz for tickers that are missing this data.

Usage:
    python update_missing_data.py                    # Update all 82 tickers
    python update_missing_data.py NVAX CERT         # Update specific tickers
    python update_missing_data.py --dry-run         # Preview without saving
"""

import sys
import time
import yfinance as yf
import pandas as pd
from datetime import date, datetime
from typing import Optional, Any, Dict, List

# Tickers that need updating (from your table - all have Fundamental=50, missing analyst data)
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
        if pd.isna(val):
            return None
        # Handle infinity
        if val == float('inf') or val == float('-inf'):
            return None
    if isinstance(val, str):
        # Try to convert string to float
        try:
            val = float(val)
            if val == float('inf') or val == float('-inf'):
                return None
        except:
            return None
    return val


def get_best_value(*values):
    """Return first non-None value."""
    for v in values:
        if v is not None and v != '' and not (isinstance(v, float) and pd.isna(v)):
            return v
    return None


def fetch_all_data(ticker: str) -> Dict[str, Any]:
    """Fetch data from yfinance + Finviz."""
    result = {
        'ticker': ticker,
        'yfinance': {},
        'finviz': {},
        'merged': {}
    }

    # === YFINANCE ===
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info or {}
        result['yfinance'] = info

        # Get recommendations
        try:
            recs = yf_ticker.recommendations
            if recs is not None and not recs.empty:
                latest = recs.iloc[-1]
                result['yfinance']['_rec_strongBuy'] = latest.get('strongBuy', 0) or 0
                result['yfinance']['_rec_buy'] = latest.get('buy', 0) or 0
                result['yfinance']['_rec_hold'] = latest.get('hold', 0) or 0
                result['yfinance']['_rec_sell'] = latest.get('sell', 0) or 0
                result['yfinance']['_rec_strongSell'] = latest.get('strongSell', 0) or 0
        except:
            pass

        # Get earnings dates
        try:
            earnings = yf_ticker.earnings_dates
            if earnings is not None and not earnings.empty:
                today = date.today()
                for idx in earnings.index:
                    dt = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                    if dt >= today:
                        result['yfinance']['_next_earnings'] = dt
                        break
        except:
            pass

    except Exception as e:
        print(f"  {Colors.RED}yfinance error: {e}{Colors.END}")

    # === FINVIZ ===
    try:
        from finvizfinance.quote import finvizfinance

        stock = finvizfinance(ticker)
        data = stock.ticker_fundament()

        if data:
            def parse_pct(val):
                if val is None or val == '-' or val == 'N/A' or val == '':
                    return None
                try:
                    return float(str(val).replace('%', '').replace(',', ''))
                except:
                    return None

            def parse_float(val):
                if val is None or val == '-' or val == 'N/A' or val == '':
                    return None
                try:
                    return float(str(val).replace(',', '').replace('$', ''))
                except:
                    return None

            result['finviz'] = {
                'inst_own': parse_pct(data.get('Inst Own')),
                'insider_own': parse_pct(data.get('Insider Own')),
                'short_float': parse_pct(data.get('Short Float')),
                'pe': parse_float(data.get('P/E')),
                'forward_pe': parse_float(data.get('Forward P/E')),
                'peg': parse_float(data.get('PEG')),
                'roe': parse_pct(data.get('ROE')),
                'roa': parse_pct(data.get('ROA')),
                'profit_margin': parse_pct(data.get('Profit Margin')),
                'revenue_growth': parse_pct(data.get('Sales Y/Y TTM')),
                'earnings_growth': parse_pct(data.get('EPS Y/Y TTM')),
                'dividend_yield': parse_pct(data.get('Dividend %')),
                'target_price': parse_float(data.get('Target Price')),
                'beta': parse_float(data.get('Beta')),
                'debt_eq': parse_float(data.get('Debt/Eq')),
                'current_ratio': parse_float(data.get('Current Ratio')),
            }

        # Get analyst ratings from Finviz
        try:
            ratings_df = stock.ticker_outer_ratings()
            if ratings_df is not None and not ratings_df.empty:
                positive = ['Buy', 'Strong Buy', 'Overweight', 'Outperform', 'Positive']
                if 'Rating' in ratings_df.columns:
                    buy_count = len(ratings_df[ratings_df['Rating'].isin(positive)])
                    result['finviz']['_ratings_positive'] = buy_count
                    result['finviz']['_ratings_total'] = len(ratings_df)
        except:
            pass

    except ImportError:
        print(f"  {Colors.YELLOW}Finviz not installed{Colors.END}")
    except Exception as e:
        print(f"  {Colors.YELLOW}Finviz error: {e}{Colors.END}")

    # === MERGE DATA ===
    yf_info = result['yfinance']
    fv = result['finviz']

    # Normalize Finviz percentages (they come as 45.5 not 0.455)
    def normalize_pct(val):
        if val is not None and abs(val) > 1:
            return val / 100
        return val

    result['merged'] = {
        # Price
        'price': get_best_value(fv.get('price'), yf_info.get('regularMarketPrice')),

        # Valuation
        'pe_ratio': get_best_value(fv.get('pe'), yf_info.get('trailingPE')),
        'forward_pe': get_best_value(fv.get('forward_pe'), yf_info.get('forwardPE')),
        'peg_ratio': get_best_value(fv.get('peg'), yf_info.get('pegRatio')),

        # Profitability (normalize Finviz %)
        'roe': normalize_pct(get_best_value(fv.get('roe'), yf_info.get('returnOnEquity'))),
        'roa': normalize_pct(get_best_value(fv.get('roa'), yf_info.get('returnOnAssets'))),
        'profit_margin': normalize_pct(get_best_value(fv.get('profit_margin'), yf_info.get('profitMargins'))),

        # Growth
        'revenue_growth': normalize_pct(get_best_value(fv.get('revenue_growth'), yf_info.get('revenueGrowth'))),
        'earnings_growth': normalize_pct(get_best_value(fv.get('earnings_growth'), yf_info.get('earningsGrowth'))),

        # Dividend
        'dividend_yield': normalize_pct(get_best_value(fv.get('dividend_yield'), yf_info.get('dividendYield'))),

        # Financial health
        'debt_equity': get_best_value(fv.get('debt_eq'), yf_info.get('debtToEquity')),
        'current_ratio': get_best_value(fv.get('current_ratio'), yf_info.get('currentRatio')),
        'beta': get_best_value(fv.get('beta'), yf_info.get('beta')),

        # Ownership (Finviz only)
        'inst_own': fv.get('inst_own'),
        'insider_own': fv.get('insider_own'),
        'short_float': fv.get('short_float'),

        # Price targets
        'target_mean': get_best_value(fv.get('target_price'), yf_info.get('targetMeanPrice')),
        'target_high': yf_info.get('targetHighPrice'),
        'target_low': yf_info.get('targetLowPrice'),

        # Analyst ratings
        'buy_count': (yf_info.get('_rec_strongBuy', 0) or 0) + (yf_info.get('_rec_buy', 0) or 0),
        'hold_count': yf_info.get('_rec_hold', 0) or 0,
        'sell_count': (yf_info.get('_rec_sell', 0) or 0) + (yf_info.get('_rec_strongSell', 0) or 0),

        # Sector/Industry
        'sector': yf_info.get('sector'),
        'industry': yf_info.get('industry'),

        # Earnings
        'next_earnings': yf_info.get('_next_earnings'),

        # Ex-dividend
        'ex_dividend_date': None,
    }

    # Handle ex-dividend date
    ex_div = yf_info.get('exDividendDate')
    if ex_div and isinstance(ex_div, (int, float)):
        result['merged']['ex_dividend_date'] = datetime.fromtimestamp(ex_div).date()

    # Calculate total ratings
    m = result['merged']
    m['total_ratings'] = m['buy_count'] + m['hold_count'] + m['sell_count']

    # Use Finviz ratings if yfinance has none
    if m['total_ratings'] == 0 and fv.get('_ratings_total', 0) > 0:
        fv_total = fv['_ratings_total']
        fv_pos = fv.get('_ratings_positive', 0)
        m['buy_count'] = fv_pos
        m['sell_count'] = int(fv_total * 0.1)
        m['hold_count'] = fv_total - fv_pos - m['sell_count']
        m['total_ratings'] = fv_total

    # Calculate analyst positivity
    if m['total_ratings'] > 0:
        m['analyst_positivity'] = m['buy_count'] / m['total_ratings'] * 100
    else:
        m['analyst_positivity'] = None

    return result


def update_database(ticker: str, data: Dict[str, Any], dry_run: bool = False) -> bool:
    """Update database tables with merged data."""
    try:
        from src.db.connection import get_connection
    except ImportError:
        print(f"  {Colors.RED}Database module not available{Colors.END}")
        return False

    m = data['merged']
    today = date.today()

    if dry_run:
        print(f"  {Colors.CYAN}[DRY RUN] Would update:{Colors.END}")
        print(f"    fundamentals: PE={m.get('pe_ratio')}, ROE={m.get('roe')}, sector={m.get('sector')}")
        print(f"    analyst_ratings: Buy={m.get('buy_count')}, Total={m.get('total_ratings')}")
        print(f"    price_targets: Mean={m.get('target_mean')}")
        return True

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # === UPDATE FUNDAMENTALS ===
                today = date.today()
                cur.execute("""
                    INSERT INTO fundamentals (ticker, date, pe_ratio, forward_pe, peg_ratio,
                                              dividend_yield, revenue_growth, earnings_growth, roe,
                                              debt_to_equity, current_ratio, sector)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, date) DO UPDATE SET
                        pe_ratio = COALESCE(EXCLUDED.pe_ratio, fundamentals.pe_ratio),
                        forward_pe = COALESCE(EXCLUDED.forward_pe, fundamentals.forward_pe),
                        peg_ratio = COALESCE(EXCLUDED.peg_ratio, fundamentals.peg_ratio),
                        dividend_yield = COALESCE(EXCLUDED.dividend_yield, fundamentals.dividend_yield),
                        revenue_growth = COALESCE(EXCLUDED.revenue_growth, fundamentals.revenue_growth),
                        earnings_growth = COALESCE(EXCLUDED.earnings_growth, fundamentals.earnings_growth),
                        roe = COALESCE(EXCLUDED.roe, fundamentals.roe),
                        debt_to_equity = COALESCE(EXCLUDED.debt_to_equity, fundamentals.debt_to_equity),
                        current_ratio = COALESCE(EXCLUDED.current_ratio, fundamentals.current_ratio),
                        sector = COALESCE(EXCLUDED.sector, fundamentals.sector)
                """, (
                    ticker, today,
                    to_native(m.get('pe_ratio')), to_native(m.get('forward_pe')), to_native(m.get('peg_ratio')),
                    to_native(m.get('dividend_yield')), to_native(m.get('revenue_growth')), to_native(m.get('earnings_growth')),
                    to_native(m.get('roe')), to_native(m.get('debt_equity')), to_native(m.get('current_ratio')),
                    m.get('sector')
                ))

                # === UPDATE ANALYST RATINGS ===
                if m.get('total_ratings', 0) > 0 or m.get('target_mean'):
                    # Check if record exists for today
                    cur.execute("SELECT 1 FROM analyst_ratings WHERE ticker = %s AND date = %s", (ticker, today))
                    exists = cur.fetchone() is not None

                    if exists:
                        cur.execute("""
                            UPDATE analyst_ratings SET
                                analyst_buy = COALESCE(%s, analyst_buy),
                                analyst_hold = COALESCE(%s, analyst_hold),
                                analyst_sell = COALESCE(%s, analyst_sell),
                                analyst_total = COALESCE(%s, analyst_total),
                                analyst_positivity = COALESCE(%s, analyst_positivity)
                            WHERE ticker = %s AND date = %s
                        """, (
                            to_native(m.get('buy_count')) if m.get('buy_count', 0) > 0 else None,
                            to_native(m.get('hold_count')) if m.get('hold_count', 0) > 0 else None,
                            to_native(m.get('sell_count')) if m.get('sell_count', 0) > 0 else None,
                            to_native(m.get('total_ratings')) if m.get('total_ratings', 0) > 0 else None,
                            to_native(m.get('analyst_positivity')),
                            ticker, today
                        ))
                    else:
                        cur.execute("""
                            INSERT INTO analyst_ratings (ticker, date, analyst_buy, analyst_hold, analyst_sell,
                                                         analyst_total, analyst_positivity)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            ticker, today,
                            to_native(m.get('buy_count')) if m.get('buy_count', 0) > 0 else None,
                            to_native(m.get('hold_count')) if m.get('hold_count', 0) > 0 else None,
                            to_native(m.get('sell_count')) if m.get('sell_count', 0) > 0 else None,
                            to_native(m.get('total_ratings')) if m.get('total_ratings', 0) > 0 else None,
                            to_native(m.get('analyst_positivity'))
                        ))

                # === UPDATE PRICE TARGETS ===
                if m.get('target_mean'):
                    cur.execute("SELECT 1 FROM price_targets WHERE ticker = %s AND date = %s", (ticker, today))
                    exists = cur.fetchone() is not None

                    if exists:
                        cur.execute("""
                            UPDATE price_targets SET
                                target_mean = COALESCE(%s, target_mean),
                                target_high = COALESCE(%s, target_high),
                                target_low = COALESCE(%s, target_low)
                            WHERE ticker = %s AND date = %s
                        """, (
                            to_native(m.get('target_mean')),
                            to_native(m.get('target_high')),
                            to_native(m.get('target_low')),
                            ticker, today
                        ))
                    else:
                        cur.execute("""
                            INSERT INTO price_targets (ticker, date, target_mean, target_high, target_low)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (
                            ticker, today,
                            to_native(m.get('target_mean')),
                            to_native(m.get('target_high')),
                            to_native(m.get('target_low'))
                        ))

                # === UPDATE PRICES ===
                if m.get('price'):
                    yf_info = data['yfinance']
                    cur.execute("""
                        INSERT INTO prices (ticker, date, close, open, high, low, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, date) DO UPDATE SET
                            close = EXCLUDED.close
                    """, (
                        ticker, today,
                        to_native(m.get('price')),
                        to_native(yf_info.get('regularMarketOpen')),
                        to_native(yf_info.get('regularMarketDayHigh')),
                        to_native(yf_info.get('regularMarketDayLow')),
                        to_native(yf_info.get('regularMarketVolume'))
                    ))

                # === UPDATE EARNINGS CALENDAR ===
                if m.get('next_earnings'):
                    try:
                        cur.execute("SELECT 1 FROM earnings_calendar WHERE ticker = %s", (ticker,))
                        exists = cur.fetchone() is not None

                        if exists:
                            cur.execute("""
                                UPDATE earnings_calendar SET earnings_date = %s WHERE ticker = %s
                            """, (m['next_earnings'], ticker))
                        else:
                            cur.execute("""
                                INSERT INTO earnings_calendar (ticker, earnings_date) VALUES (%s, %s)
                            """, (ticker, m['next_earnings']))
                    except Exception as e:
                        pass  # Table might not exist

            conn.commit()
        return True

    except Exception as e:
        print(f"  {Colors.RED}Database error: {e}{Colors.END}")
        return False


def process_ticker(ticker: str, dry_run: bool = False) -> Dict:
    """Process a single ticker."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}[{ticker}]{Colors.END}")

    # Fetch data
    data = fetch_all_data(ticker)
    m = data['merged']

    # Show what we found
    fields_found = []
    if m.get('pe_ratio'):
        try:
            fields_found.append(f"PE={float(m['pe_ratio']):.1f}")
        except:
            fields_found.append(f"PE={m['pe_ratio']}")
    if m.get('roe'):
        try:
            fields_found.append(f"ROE={float(m['roe']):.1%}")
        except:
            fields_found.append(f"ROE={m['roe']}")
    if m.get('target_mean'):
        try:
            fields_found.append(f"Target=${float(m['target_mean']):.2f}")
        except:
            fields_found.append(f"Target={m['target_mean']}")
    if m.get('total_ratings', 0) > 0: fields_found.append(f"Analysts={m['total_ratings']}")
    if m.get('inst_own'):
        try:
            fields_found.append(f"Inst={float(m['inst_own']):.1f}%")
        except:
            fields_found.append(f"Inst={m['inst_own']}%")
    if m.get('next_earnings'): fields_found.append(f"Earn={m['next_earnings']}")

    if fields_found:
        print(f"  Found: {', '.join(fields_found)}")
    else:
        print(f"  {Colors.YELLOW}No new data found{Colors.END}")
        return {'ticker': ticker, 'success': False, 'reason': 'No data'}

    # Update database
    if update_database(ticker, data, dry_run):
        print(f"  {Colors.GREEN}✓ Updated{Colors.END}")
        return {'ticker': ticker, 'success': True, 'data': m}
    else:
        return {'ticker': ticker, 'success': False, 'reason': 'DB error'}


def main():
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}Update Missing Data - yfinance + Finviz{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")

    # Parse arguments
    tickers = []
    dry_run = False

    for arg in sys.argv[1:]:
        if arg.lower() == '--dry-run':
            dry_run = True
        elif arg.upper() in [t.upper() for t in TICKERS_TO_UPDATE] or len(arg) <= 5:
            tickers.append(arg.upper())

    # Use provided tickers or default list
    if not tickers:
        tickers = TICKERS_TO_UPDATE
        print(f"\nUpdating all {len(tickers)} tickers from list")
    else:
        print(f"\nUpdating {len(tickers)} specified tickers: {', '.join(tickers)}")

    if dry_run:
        print(f"{Colors.YELLOW}DRY RUN MODE - No changes will be saved{Colors.END}")

    # Check Finviz availability
    try:
        from finvizfinance.quote import finvizfinance
        print(f"{Colors.GREEN}✓ Finviz available{Colors.END}")
    except ImportError:
        print(f"{Colors.YELLOW}⚠ Finviz not installed - using yfinance only{Colors.END}")

    # Process tickers
    results = []
    success_count = 0

    for i, ticker in enumerate(tickers):
        print(f"\n[{i+1}/{len(tickers)}]", end="")
        result = process_ticker(ticker, dry_run)
        results.append(result)
        if result['success']:
            success_count += 1

        # Rate limit
        if i < len(tickers) - 1:
            time.sleep(0.5)

    # Summary
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}SUMMARY{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"Total: {len(tickers)}")
    print(f"{Colors.GREEN}Success: {success_count}{Colors.END}")
    print(f"{Colors.RED}Failed: {len(tickers) - success_count}{Colors.END}")

    if not dry_run and success_count > 0:
        print(f"\n{Colors.GREEN}✓ Database updated! Reload signals table to see changes.{Colors.END}")


if __name__ == "__main__":
    main()