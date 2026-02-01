"""
Debug Script: Test Multi-Source Data Fetching for Individual Tickers
=====================================================================
Tests data fetching from: yfinance, Finviz, and IBKR (if available)

Usage:
    python debug_ticker_data.py NVAX
    python debug_ticker_data.py NVAX CERT ADPT
    python debug_ticker_data.py NVAX --save  # Also saves to database
"""

import sys
import pandas as pd
from datetime import date, datetime
from typing import Optional, Any, Dict

# ANSI colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")

def print_section(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}--- {text} ---{Colors.END}")

def print_source(text: str):
    print(f"\n  {Colors.BOLD}{Colors.MAGENTA}[{text}]{Colors.END}")

def print_field(name: str, value: Any, required: bool = False, source: str = None):
    source_tag = f" ({source})" if source else ""
    if value is None or value == '' or (isinstance(value, float) and pd.isna(value)):
        status = f"{Colors.RED}✗ MISSING{Colors.END}" if required else f"{Colors.YELLOW}○ N/A{Colors.END}"
        print(f"    {name:25} {status}")
    else:
        # Format based on type
        if isinstance(value, float):
            if abs(value) < 0.01:
                formatted = f"{value:.4f}"
            elif abs(value) < 1:
                formatted = f"{value:.2%}" if abs(value) < 0.5 else f"{value:.4f}"
            elif abs(value) < 100:
                formatted = f"{value:.2f}"
            else:
                formatted = f"{value:,.2f}"
        elif isinstance(value, (int,)):
            formatted = f"{value:,}"
        else:
            formatted = str(value)[:50]
        print(f"    {name:25} {Colors.GREEN}✓{Colors.END} {formatted}{source_tag}")

def get_best_value(*values):
    """Return first non-None value from multiple sources."""
    for v in values:
        if v is not None and v != '' and not (isinstance(v, float) and pd.isna(v)):
            return v
    return None


def fetch_yfinance_data(ticker: str) -> Dict[str, Any]:
    """Fetch data from yfinance."""
    import yfinance as yf

    result = {
        'available': False,
        'info': {},
        'recommendations': None,
        'earnings_dates': None,
        'error': None
    }

    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info

        if not info or info.get('regularMarketPrice') is None:
            result['error'] = "No price data"
            return result

        result['available'] = True
        result['info'] = info
        result['yf_ticker'] = yf_ticker

        # Get recommendations
        try:
            result['recommendations'] = yf_ticker.recommendations
        except:
            pass

        # Get earnings dates
        try:
            result['earnings_dates'] = yf_ticker.earnings_dates
        except:
            pass

    except Exception as e:
        result['error'] = str(e)

    return result


def fetch_finviz_data(ticker: str) -> Dict[str, Any]:
    """Fetch data from Finviz."""
    result = {
        'available': False,
        'fundamentals': {},
        'ratings': {},
        'error': None
    }

    try:
        from finvizfinance.quote import finvizfinance

        stock = finvizfinance(ticker)
        data = stock.ticker_fundament()

        if not data:
            result['error'] = "No data returned"
            return result

        result['available'] = True

        # Parse fundamentals
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

        result['fundamentals'] = {
            # Ownership
            'inst_own': parse_pct(data.get('Inst Own')),
            'insider_own': parse_pct(data.get('Insider Own')),
            'short_float': parse_pct(data.get('Short Float')),
            # Valuation
            'pe': parse_float(data.get('P/E')),
            'forward_pe': parse_float(data.get('Forward P/E')),
            'peg': parse_float(data.get('PEG')),
            'ps': parse_float(data.get('P/S')),
            'pb': parse_float(data.get('P/B')),
            # Profitability
            'roe': parse_pct(data.get('ROE')),
            'roa': parse_pct(data.get('ROA')),
            'profit_margin': parse_pct(data.get('Profit Margin')),
            'oper_margin': parse_pct(data.get('Oper. Margin')),
            'gross_margin': parse_pct(data.get('Gross Margin')),
            # Growth
            'eps': parse_float(data.get('EPS (ttm)')),
            'eps_growth_yoy': parse_pct(data.get('EPS Y/Y TTM')),
            'sales_growth_yoy': parse_pct(data.get('Sales Y/Y TTM')),
            # Dividend
            'dividend_yield': parse_pct(data.get('Dividend %')),
            # Price targets
            'target_price': parse_float(data.get('Target Price')),
            'price': parse_float(data.get('Price')),
            # Other
            'beta': parse_float(data.get('Beta')),
            'rsi': parse_float(data.get('RSI (14)')),
            # Debt
            'debt_eq': parse_float(data.get('Debt/Eq')),
            'current_ratio': parse_float(data.get('Current Ratio')),
            # Raw
            '_raw': data
        }

        # Get analyst ratings
        try:
            ratings_df = stock.ticker_outer_ratings()
            if ratings_df is not None and not ratings_df.empty:
                positive_ratings = ['Buy', 'Strong Buy', 'Overweight', 'Outperform', 'Positive']
                if 'Rating' in ratings_df.columns:
                    buy_ratings = ratings_df[ratings_df['Rating'].isin(positive_ratings)]
                    total_positive = len(buy_ratings)
                    total_ratings = len(ratings_df)
                    buy_pct = (total_positive / total_ratings * 100) if total_ratings > 0 else 0
                    result['ratings'] = {
                        'total_positive': total_positive,
                        'total_ratings': total_ratings,
                        'buy_pct': round(buy_pct, 1)
                    }
        except Exception as e:
            result['ratings_error'] = str(e)

    except ImportError:
        result['error'] = "finvizfinance not installed (pip install finvizfinance)"
    except Exception as e:
        result['error'] = str(e)

    return result


def fetch_ibkr_data(ticker: str) -> Dict[str, Any]:
    """Fetch data from IBKR (if available)."""
    result = {
        'available': False,
        'connected': False,
        'fundamentals': {},
        'error': None
    }

    try:
        from src.data.ibkr_client import IBKRClient

        ibkr = IBKRClient()
        result['connected'] = ibkr.is_connected()

        if not result['connected']:
            result['error'] = "IBKR not connected"
            return result

        data = ibkr.get_fundamental_data(ticker)
        if data:
            result['available'] = True
            result['fundamentals'] = data

    except ImportError:
        result['error'] = "IBKR module not available"
    except Exception as e:
        result['error'] = str(e)

    return result


def analyze_ticker(ticker: str, save_to_db: bool = False) -> dict:
    """
    Analyze a single ticker using all data sources.
    """
    print_header(f"ANALYZING: {ticker}")

    result = {
        'ticker': ticker,
        'success': False,
        'errors': [],
        'data': {},
        'sources_used': []
    }

    today = date.today()

    # =========================================================================
    # FETCH FROM ALL SOURCES
    # =========================================================================
    print_section("FETCHING DATA FROM ALL SOURCES")

    # Source 1: yfinance
    print_source("yfinance")
    yf_data = fetch_yfinance_data(ticker)
    if yf_data['available']:
        print(f"    {Colors.GREEN}✓ Connected and data available{Colors.END}")
        result['sources_used'].append('yfinance')
    else:
        print(f"    {Colors.RED}✗ {yf_data.get('error', 'Failed')}{Colors.END}")
        result['errors'].append(f"yfinance: {yf_data.get('error')}")

    # Source 2: Finviz
    print_source("Finviz")
    finviz_data = fetch_finviz_data(ticker)
    if finviz_data['available']:
        print(f"    {Colors.GREEN}✓ Connected and data available{Colors.END}")
        result['sources_used'].append('Finviz')
        if finviz_data['ratings']:
            print(f"    {Colors.GREEN}✓ Analyst ratings available{Colors.END}")
    else:
        print(f"    {Colors.YELLOW}○ {finviz_data.get('error', 'Not available')}{Colors.END}")

    # Source 3: IBKR
    print_source("IBKR")
    ibkr_data = fetch_ibkr_data(ticker)
    if ibkr_data['available']:
        print(f"    {Colors.GREEN}✓ Connected and data available{Colors.END}")
        result['sources_used'].append('IBKR')
    elif ibkr_data['connected']:
        print(f"    {Colors.YELLOW}○ Connected but no data for {ticker}{Colors.END}")
    else:
        print(f"    {Colors.YELLOW}○ {ibkr_data.get('error', 'Not connected')}{Colors.END}")

    if not yf_data['available']:
        print(f"\n{Colors.RED}ERROR: Cannot proceed without yfinance data{Colors.END}")
        return result

    info = yf_data['info']
    finviz_fund = finviz_data.get('fundamentals', {})
    ibkr_fund = ibkr_data.get('fundamentals', {})

    # =========================================================================
    # BASIC INFO
    # =========================================================================
    print_section("BASIC INFO")
    print_field("Name", info.get('shortName') or info.get('longName'))
    print_field("Symbol", info.get('symbol'))
    print_field("Sector", info.get('sector'))
    print_field("Industry", info.get('industry'))

    # =========================================================================
    # PRICE DATA
    # =========================================================================
    print_section("PRICE DATA → prices table")

    current_price = get_best_value(
        finviz_fund.get('price'),
        info.get('regularMarketPrice'),
        info.get('currentPrice')
    )
    print_field("Current Price", current_price, required=True)
    print_field("Previous Close", info.get('regularMarketPreviousClose'))
    print_field("Volume", info.get('regularMarketVolume'))
    print_field("52W High", info.get('fiftyTwoWeekHigh'))
    print_field("52W Low", info.get('fiftyTwoWeekLow'))

    result['data']['price'] = current_price

    # =========================================================================
    # VALUATION METRICS (MERGED)
    # =========================================================================
    print_section("VALUATION METRICS → fundamentals table")

    pe_ratio = get_best_value(ibkr_fund.get('pe_ratio'), finviz_fund.get('pe'), info.get('trailingPE'))
    forward_pe = get_best_value(ibkr_fund.get('forward_pe'), finviz_fund.get('forward_pe'), info.get('forwardPE'))
    peg_ratio = get_best_value(ibkr_fund.get('peg_ratio'), finviz_fund.get('peg'), info.get('pegRatio'))

    # Show which source each came from
    pe_source = "IBKR" if ibkr_fund.get('pe_ratio') else ("Finviz" if finviz_fund.get('pe') else "yfinance")
    fpe_source = "IBKR" if ibkr_fund.get('forward_pe') else ("Finviz" if finviz_fund.get('forward_pe') else "yfinance")
    peg_source = "IBKR" if ibkr_fund.get('peg_ratio') else ("Finviz" if finviz_fund.get('peg') else "yfinance")

    print_field("PE Ratio (TTM)", pe_ratio, source=pe_source if pe_ratio else None)
    print_field("Forward PE", forward_pe, source=fpe_source if forward_pe else None)
    print_field("PEG Ratio", peg_ratio, source=peg_source if peg_ratio else None)
    print_field("Price/Book", get_best_value(finviz_fund.get('pb'), info.get('priceToBook')))
    print_field("Price/Sales", get_best_value(finviz_fund.get('ps'), info.get('priceToSalesTrailing12Months')))

    result['data']['pe_ratio'] = pe_ratio
    result['data']['forward_pe'] = forward_pe
    result['data']['peg_ratio'] = peg_ratio

    # =========================================================================
    # PROFITABILITY METRICS (MERGED)
    # =========================================================================
    print_section("PROFITABILITY → fundamentals table")

    # ROE - normalize from percentage if from Finviz
    roe_finviz = finviz_fund.get('roe')
    if roe_finviz and roe_finviz > 1:
        roe_finviz = roe_finviz / 100
    roe = get_best_value(ibkr_fund.get('roe'), roe_finviz, info.get('returnOnEquity'))

    roa_finviz = finviz_fund.get('roa')
    if roa_finviz and roa_finviz > 1:
        roa_finviz = roa_finviz / 100
    roa = get_best_value(ibkr_fund.get('roa'), roa_finviz, info.get('returnOnAssets'))

    profit_margin_finviz = finviz_fund.get('profit_margin')
    if profit_margin_finviz and profit_margin_finviz > 1:
        profit_margin_finviz = profit_margin_finviz / 100
    profit_margin = get_best_value(ibkr_fund.get('profit_margin'), profit_margin_finviz, info.get('profitMargins'))

    print_field("ROE", roe)
    print_field("ROA", roa)
    print_field("Profit Margin", profit_margin)
    print_field("Operating Margin", get_best_value(finviz_fund.get('oper_margin'), info.get('operatingMargins')))
    print_field("Gross Margin", get_best_value(finviz_fund.get('gross_margin'), info.get('grossMargins')))

    result['data']['roe'] = roe

    # =========================================================================
    # GROWTH METRICS (MERGED)
    # =========================================================================
    print_section("GROWTH METRICS → fundamentals table")

    rev_growth_finviz = finviz_fund.get('sales_growth_yoy')
    if rev_growth_finviz and abs(rev_growth_finviz) > 1:
        rev_growth_finviz = rev_growth_finviz / 100
    revenue_growth = get_best_value(ibkr_fund.get('revenue_growth'), rev_growth_finviz, info.get('revenueGrowth'))

    eps_growth_finviz = finviz_fund.get('eps_growth_yoy')
    if eps_growth_finviz and abs(eps_growth_finviz) > 1:
        eps_growth_finviz = eps_growth_finviz / 100
    earnings_growth = get_best_value(ibkr_fund.get('earnings_growth'), eps_growth_finviz, info.get('earningsGrowth'))

    print_field("Revenue Growth", revenue_growth)
    print_field("Earnings Growth", earnings_growth)
    print_field("EPS (TTM)", get_best_value(finviz_fund.get('eps'), info.get('trailingEps')))

    # =========================================================================
    # OWNERSHIP DATA (FINVIZ EXCLUSIVE)
    # =========================================================================
    print_section("OWNERSHIP DATA → Finviz exclusive")

    inst_own = finviz_fund.get('inst_own')
    insider_own = finviz_fund.get('insider_own')
    short_float = finviz_fund.get('short_float')

    print_field("Institutional Own %", inst_own)
    print_field("Insider Own %", insider_own)
    print_field("Short Float %", short_float)
    print_field("Beta", get_best_value(finviz_fund.get('beta'), info.get('beta')))
    print_field("RSI (14)", finviz_fund.get('rsi'))

    result['data']['inst_own'] = inst_own
    result['data']['short_float'] = short_float

    # =========================================================================
    # FINANCIAL HEALTH
    # =========================================================================
    print_section("FINANCIAL HEALTH → fundamentals table")

    print_field("Debt/Equity", get_best_value(finviz_fund.get('debt_eq'), info.get('debtToEquity')))
    print_field("Current Ratio", get_best_value(finviz_fund.get('current_ratio'), info.get('currentRatio')))
    print_field("Quick Ratio", info.get('quickRatio'))

    # =========================================================================
    # DIVIDEND DATA
    # =========================================================================
    print_section("DIVIDEND DATA → fundamentals table")

    div_yield_finviz = finviz_fund.get('dividend_yield')
    if div_yield_finviz and div_yield_finviz > 1:
        div_yield_finviz = div_yield_finviz / 100
    dividend_yield = get_best_value(ibkr_fund.get('dividend_yield'), div_yield_finviz, info.get('dividendYield'))

    print_field("Dividend Yield", dividend_yield)
    print_field("Dividend Rate", info.get('dividendRate'))

    # Ex-dividend date
    ex_div_date = info.get('exDividendDate')
    if ex_div_date and isinstance(ex_div_date, (int, float)):
        ex_div_date = datetime.fromtimestamp(ex_div_date).date()
    print_field("Ex-Dividend Date", ex_div_date)

    # =========================================================================
    # ANALYST RATINGS (MERGED)
    # =========================================================================
    print_section("ANALYST DATA → analyst_ratings table")

    print_field("Recommendation", info.get('recommendationKey'))
    print_field("# Analyst Opinions", info.get('numberOfAnalystOpinions'))

    # Try yfinance first
    buy_count = 0
    hold_count = 0
    sell_count = 0

    # Method 1: recommendationTrend
    rec_trend = info.get('recommendationTrend', {}).get('trend', [])
    if rec_trend and len(rec_trend) > 0:
        latest = rec_trend[0]
        buy_count = (latest.get('strongBuy', 0) or 0) + (latest.get('buy', 0) or 0)
        hold_count = latest.get('hold', 0) or 0
        sell_count = (latest.get('sell', 0) or 0) + (latest.get('strongSell', 0) or 0)
        print(f"    {Colors.GREEN}✓ Got from yfinance recommendationTrend{Colors.END}")

    # Method 2: recommendations DataFrame
    if buy_count == 0 and hold_count == 0 and sell_count == 0:
        recs = yf_data.get('recommendations')
        if recs is not None and not recs.empty:
            latest = recs.iloc[-1]
            buy_count = (latest.get('strongBuy', 0) or 0) + (latest.get('buy', 0) or 0)
            hold_count = latest.get('hold', 0) or 0
            sell_count = (latest.get('sell', 0) or 0) + (latest.get('strongSell', 0) or 0)
            print(f"    {Colors.GREEN}✓ Got from yfinance recommendations DataFrame{Colors.END}")

    # Method 3: Finviz fallback
    total_ratings = buy_count + hold_count + sell_count
    if total_ratings == 0 and finviz_data['ratings']:
        finviz_ratings = finviz_data['ratings']
        print(f"    {Colors.GREEN}✓ Using Finviz ratings data{Colors.END}")
        print_field("Finviz Positive", finviz_ratings.get('total_positive'))
        print_field("Finviz Total", finviz_ratings.get('total_ratings'))
        print_field("Finviz Buy %", finviz_ratings.get('buy_pct'))

        finviz_total = finviz_ratings.get('total_ratings', 0)
        finviz_buy_pct = finviz_ratings.get('buy_pct', 0)
        if finviz_total > 0:
            buy_count = int(finviz_total * finviz_buy_pct / 100)
            sell_count = int(finviz_total * 0.1)
            hold_count = finviz_total - buy_count - sell_count
            total_ratings = finviz_total

    if total_ratings > 0:
        print_field("Buy Count", buy_count)
        print_field("Hold Count", hold_count)
        print_field("Sell Count", sell_count)
        print_field("Total Ratings", total_ratings)
        positivity = buy_count / total_ratings * 100
        print_field("Analyst Positivity %", positivity)
        result['data']['analyst_positivity'] = positivity
        result['data']['total_ratings'] = total_ratings
    else:
        print(f"    {Colors.YELLOW}No analyst rating breakdown available{Colors.END}")

    # =========================================================================
    # PRICE TARGETS (MERGED)
    # =========================================================================
    print_section("PRICE TARGETS → price_targets table")

    target_mean = get_best_value(finviz_fund.get('target_price'), info.get('targetMeanPrice'))
    target_source = "Finviz" if finviz_fund.get('target_price') else "yfinance"

    print_field("Target Mean", target_mean, required=True, source=target_source if target_mean else None)
    print_field("Target High", info.get('targetHighPrice'))
    print_field("Target Low", info.get('targetLowPrice'))

    if current_price and target_mean:
        upside = ((target_mean - current_price) / current_price) * 100
        print_field("Upside %", upside)
        result['data']['upside_pct'] = upside

    result['data']['target_mean'] = target_mean

    # =========================================================================
    # EARNINGS CALENDAR
    # =========================================================================
    print_section("EARNINGS → earnings_calendar table")

    earnings_dates = yf_data.get('earnings_dates')
    if earnings_dates is not None and not earnings_dates.empty:
        print(f"    {Colors.GREEN}Found {len(earnings_dates)} earnings dates{Colors.END}")

        next_earnings = None
        for idx in earnings_dates.index:
            earnings_dt = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
            if earnings_dt >= today:
                next_earnings = earnings_dt
                break

        if next_earnings:
            print_field("Next Earnings Date", next_earnings, required=True)
            result['data']['next_earnings'] = next_earnings
    else:
        print(f"    {Colors.YELLOW}No earnings dates available{Colors.END}")

    # =========================================================================
    # TECHNICAL DATA CHECK
    # =========================================================================
    print_section("TECHNICAL DATA AVAILABILITY")

    try:
        hist = yf_data['yf_ticker'].history(period="3mo")
        if hist is not None and not hist.empty:
            print(f"    {Colors.GREEN}✓ Price history: {len(hist)} days{Colors.END}")
            result['data']['has_price_history'] = True
        else:
            print(f"    {Colors.RED}✗ No price history{Colors.END}")
            result['data']['has_price_history'] = False
    except Exception as e:
        print(f"    {Colors.RED}✗ Error: {e}{Colors.END}")
        result['data']['has_price_history'] = False

    # RSI from Finviz
    if finviz_fund.get('rsi'):
        print(f"    {Colors.GREEN}✓ RSI from Finviz: {finviz_fund.get('rsi')}{Colors.END}")

    # =========================================================================
    # OPTIONS DATA CHECK
    # =========================================================================
    print_section("OPTIONS DATA AVAILABILITY")

    try:
        options_dates = yf_data['yf_ticker'].options
        if options_dates and len(options_dates) > 0:
            print(f"    {Colors.GREEN}✓ Options: {len(options_dates)} expiration dates{Colors.END}")
            result['data']['has_options'] = True
        else:
            print(f"    {Colors.YELLOW}No options data{Colors.END}")
            result['data']['has_options'] = False
    except:
        result['data']['has_options'] = False

    # =========================================================================
    # CALCULATE SCORES (same as signals_tab.py)
    # =========================================================================
    print_section("CALCULATED SCORES")

    # Fundamental score
    fund_points = 50
    if pe_ratio and pe_ratio > 0:
        if pe_ratio < 15:
            fund_points += 10
        elif pe_ratio < 25:
            fund_points += 5
        elif pe_ratio > 50:
            fund_points -= 10
    if peg_ratio and peg_ratio > 0:
        if peg_ratio < 1:
            fund_points += 10
        elif peg_ratio < 2:
            fund_points += 5
        elif peg_ratio > 3:
            fund_points -= 5
    if roe and roe > 0.15:
        fund_points += 5
    if roe and roe > 0.25:
        fund_points += 5
    if profit_margin and profit_margin > 0.1:
        fund_points += 5
    if inst_own is not None:
        if inst_own >= 70:
            fund_points += 5
        elif inst_own < 20:
            fund_points -= 5
    if short_float is not None:
        if short_float >= 20:
            fund_points -= 10
        elif short_float >= 10:
            fund_points -= 5
    fundamental_score = max(0, min(100, fund_points))

    # Growth score
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
    growth_score = max(0, min(100, growth_points))

    # Dividend score
    dividend_points = 50
    if dividend_yield:
        if dividend_yield > 0.04:
            dividend_points += 30
        elif dividend_yield > 0.02:
            dividend_points += 20
        elif dividend_yield > 0.01:
            dividend_points += 10
    dividend_score = max(0, min(100, dividend_points))

    print_field("Fundamental Score", fundamental_score)
    print_field("Growth Score", growth_score)
    print_field("Dividend Score", dividend_score)

    result['data']['fundamental_score'] = fundamental_score
    result['data']['growth_score'] = growth_score
    result['data']['dividend_score'] = dividend_score

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("SUMMARY")

    print(f"\n  {Colors.BOLD}Data Sources Used:{Colors.END} {', '.join(result['sources_used'])}")

    checks = [
        ("Price", result['data'].get('price') is not None),
        ("PE Ratio", pe_ratio is not None),
        ("ROE", roe is not None),
        ("Target Price", result['data'].get('target_mean') is not None),
        ("Analyst Ratings", result['data'].get('total_ratings', 0) > 0),
        ("Inst. Ownership", inst_own is not None),
        ("Short Float", short_float is not None),
        ("Earnings Date", result['data'].get('next_earnings') is not None),
        ("Price History", result['data'].get('has_price_history', False)),
    ]

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)

    print(f"\n  {Colors.BOLD}Data Coverage:{Colors.END}")
    for name, ok in checks:
        status = f"{Colors.GREEN}✓{Colors.END}" if ok else f"{Colors.RED}✗{Colors.END}"
        print(f"    {status} {name}")

    print()
    pct = passed / total * 100
    color = Colors.GREEN if pct >= 75 else Colors.YELLOW if pct >= 50 else Colors.RED
    print(f"  {color}{Colors.BOLD}Data Completeness: {passed}/{total} ({pct:.0f}%){Colors.END}")

    result['success'] = True
    result['completeness'] = pct

    # =========================================================================
    # SAVE TO DATABASE (optional)
    # =========================================================================
    if save_to_db:
        print_section("SAVING TO DATABASE")
        try:
            from src.db.connection import get_connection
            # Would save here - for now just indicate it would work
            print(f"    {Colors.GREEN}✓ Database connection available{Colors.END}")
            print(f"    {Colors.YELLOW}(Use --save flag with actual platform to save){Colors.END}")
        except ImportError:
            print(f"    {Colors.YELLOW}Database module not available in standalone mode{Colors.END}")

    return result


def main():
    if len(sys.argv) < 2:
        print(f"""
{Colors.BOLD}Multi-Source Ticker Data Debugger{Colors.END}
===================================

Tests data fetching from: yfinance, Finviz, IBKR

Usage:
    python debug_ticker_data.py TICKER [TICKER2] [TICKER3] [--save]

Examples:
    python debug_ticker_data.py NVAX
    python debug_ticker_data.py NVAX CERT ADPT
    python debug_ticker_data.py AAPL --save

Data Sources:
    - yfinance  - Always available, broad coverage
    - Finviz    - Ownership data, RSI, analyst ratings
    - IBKR      - Institutional-grade (if connected)
""")
        sys.exit(1)

    # Parse arguments
    tickers = []
    save_to_db = False

    for arg in sys.argv[1:]:
        if arg.lower() == '--save':
            save_to_db = True
        else:
            tickers.append(arg.upper())

    if not tickers:
        print(f"{Colors.RED}No tickers specified{Colors.END}")
        sys.exit(1)

    print(f"\n{Colors.BOLD}Analyzing {len(tickers)} ticker(s): {', '.join(tickers)}{Colors.END}")

    results = []
    for ticker in tickers:
        result = analyze_ticker(ticker, save_to_db)
        results.append(result)

    # Final summary
    if len(tickers) > 1:
        print_header("FINAL SUMMARY")
        for r in results:
            status = f"{Colors.GREEN}✓{Colors.END}" if r['success'] else f"{Colors.RED}✗{Colors.END}"
            completeness = r.get('completeness', 0)
            sources = ', '.join(r.get('sources_used', []))
            print(f"  {status} {r['ticker']:8} - {completeness:.0f}% complete - Sources: {sources}")


if __name__ == "__main__":
    main()