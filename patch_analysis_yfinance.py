"""
Patch remaining direct yfinance calls in analysis.py:
1. Add get_earnings_history() to yf_subprocess.py
2. Patch _check_earnings_status() to use subprocess
3. Patch _run_earnings_aware_analysis() to use subprocess
"""

import ast

# ============================================================================
# PATCH 1: Add get_earnings_history() to yf_subprocess.py
# ============================================================================

with open('src/analytics/yf_subprocess.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the end of the file and add the new function
new_function = '''

def get_earnings_history(ticker: str, timeout: int = 8) -> Optional[pd.DataFrame]:
    """
    Get earnings history (EPS actual vs estimate) via subprocess.
    Returns DataFrame with epsActual, epsEstimate, epsDifference, surprisePercent.
    """
    script = f"""
import json, yfinance as yf, traceback
try:
    stock = yf.Ticker("{ticker}")
    eh = stock.earnings_history
    if eh is not None and not eh.empty:
        records = []
        for idx, row in eh.iterrows():
            rec = {{"date": str(idx)[:10]}}
            for col in eh.columns:
                val = row[col]
                if val is not None and str(val) != 'nan':
                    try:
                        rec[col] = float(val)
                    except (ValueError, TypeError):
                        rec[col] = str(val)
            records.append(rec)
        print(json.dumps({{"ok": True, "records": records}}))
    else:
        print(json.dumps({{"ok": True, "records": []}}))
except Exception as e:
    print(json.dumps({{"ok": False, "error": str(e)}}))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout, check=False
        )
        if result.stdout.strip():
            payload = json.loads(result.stdout.strip())
            if payload.get("ok") and payload.get("records"):
                df = pd.DataFrame(payload["records"])
                if "date" in df.columns:
                    df.index = pd.to_datetime(df["date"])
                    df = df.drop(columns=["date"])
                return df
    except subprocess.TimeoutExpired:
        logger.warning(f"{ticker}: get_earnings_history timed out after {timeout}s")
    except Exception as e:
        logger.debug(f"{ticker}: get_earnings_history error: {e}")
    return None
'''

if 'def get_earnings_history' not in content:
    content = content.rstrip() + '\n' + new_function
    print("PATCH 1: Added get_earnings_history() to yf_subprocess.py")
else:
    print("PATCH 1: get_earnings_history() already exists")

with open('src/analytics/yf_subprocess.py', 'w', encoding='utf-8') as f:
    f.write(content)


# ============================================================================
# PATCH 2: Fix _check_earnings_status() in analysis.py
# ============================================================================

with open('src/tabs/signals_tab/analysis.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_check_earnings = '''def _check_earnings_status(ticker: str) -> str:
    """Check if ticker is near earnings. Returns 'pre', 'post', or 'none'."""
    try:
        import yfinance as yf

        engine = get_engine()

        # Check database for earnings date
        df = pd.read_sql(f"""
            SELECT earnings_date FROM earnings_calendar 
            WHERE ticker = '{ticker}' 
            AND earnings_date >= CURRENT_DATE - INTERVAL '5 days'
            AND earnings_date <= CURRENT_DATE + INTERVAL '5 days'
            ORDER BY ABS(earnings_date - CURRENT_DATE) LIMIT 1
        """, engine)

        if not df.empty and df.iloc[0]['earnings_date']:
            ed = pd.to_datetime(df.iloc[0]['earnings_date']).date()
            days = (ed - date.today()).days

            if days < 0:
                return 'post'
            elif days <= 5:
                return 'pre'

        # Fallback to yfinance
        stock = yf.Ticker(ticker)
        try:
            hist = stock.earnings_history
            if hist is not None and not hist.empty:
                latest_date = pd.to_datetime(hist.index[0]).date()
                days = (latest_date - date.today()).days
                if days >= -5 and days <= 0:
                    return 'post'
        except:
            pass

    except Exception as e:
        logger.debug(f"Earnings status check error for {ticker}: {e}")

    return 'none\''''

new_check_earnings = '''def _check_earnings_status(ticker: str) -> str:
    """Check if ticker is near earnings. Returns 'pre', 'post', or 'none'."""
    try:
        engine = get_engine()

        # Check database for earnings date
        df = pd.read_sql(f"""
            SELECT earnings_date FROM earnings_calendar 
            WHERE ticker = '{ticker}' 
            AND earnings_date >= CURRENT_DATE - INTERVAL '5 days'
            AND earnings_date <= CURRENT_DATE + INTERVAL '5 days'
            ORDER BY ABS(earnings_date - CURRENT_DATE) LIMIT 1
        """, engine)

        if not df.empty and df.iloc[0]['earnings_date']:
            ed = pd.to_datetime(df.iloc[0]['earnings_date']).date()
            days = (ed - date.today()).days

            if days < 0:
                return 'post'
            elif days <= 5:
                return 'pre'

        # Fallback to yfinance via subprocess (safe from Streamlit freeze)
        from src.analytics.yf_subprocess import get_earnings_history
        hist = get_earnings_history(ticker)
        if hist is not None and not hist.empty:
            try:
                latest_date = pd.to_datetime(hist.index[0]).date()
                days = (latest_date - date.today()).days
                if days >= -5 and days <= 0:
                    return 'post'
            except Exception:
                pass

    except Exception as e:
        logger.debug(f"Earnings status check error for {ticker}: {e}")

    return 'none\''''

if old_check_earnings in content:
    content = content.replace(old_check_earnings, new_check_earnings)
    print("PATCH 2: Fixed _check_earnings_status() to use subprocess")
else:
    print("PATCH 2: WARNING — Could not find _check_earnings_status()")


# ============================================================================
# PATCH 3: Fix _run_earnings_aware_analysis() in analysis.py
# ============================================================================

old_run_earnings = '''def _run_earnings_aware_analysis(ticker: str, earnings_status: str) -> Dict:
    """Run analysis with earnings context."""
    import yfinance as yf

    result = {
        'ticker': ticker,
        'news_count': 0,
        'sentiment_score': None,
        'earnings_summary': None,
    }

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get('shortName', info.get('longName', ticker))

        # Build earnings-specific search queries
        if earnings_status == 'post':
            queries = [
                f"{ticker} earnings results",
                f"{company_name} quarterly earnings",
                f"{ticker} earnings reaction",
                f"{ticker} guidance outlook",
            ]
        else:
            queries = [
                f"{ticker} earnings preview expectations",
                f"{company_name} earnings whisper",
            ]

        nc = NewsCollector()
        all_articles = []

        # Collect with earnings queries
        for query in queries:
            try:
                articles = nc.collect_ai_search(ticker, company_name=query)
                for a in articles:
                    a['ticker'] = ticker
                all_articles.extend(articles)
            except:
                pass

        # Also standard collection with force refresh
        standard = nc.collect_and_save(ticker, days_back=5, force_refresh=True)
        all_articles.extend(standard.get('articles', []))

        # Deduplicate
        seen = set()
        unique = []
        for a in all_articles:
            title = str(a.get('title', '')).lower()[:40]
            if title and title not in seen:
                seen.add(title)
                unique.append(a)

        result['news_count'] = len(unique)

        # Save and analyze sentiment
        if unique:
            nc.save_articles(unique)

            sa = SentimentAnalyzer()
            sent_result = sa.analyze_ticker_sentiment(ticker, unique)
            if sent_result:
                result['sentiment_score'] = sent_result.get('sentiment_score')

        # Get earnings result if post-earnings
        if earnings_status == 'post':
            try:
                hist = stock.earnings_history
                if hist is not None and not hist.empty:
                    latest = hist.iloc[0]
                    eps_actual = latest.get('epsActual')
                    eps_est = latest.get('epsEstimate')

                    surprise_pct = None
                    if eps_actual and eps_est and eps_est != 0:
                        surprise_pct = ((eps_actual - eps_est) / abs(eps_est)) * 100

                    # Get price reaction
                    price_hist = stock.history(period="5d")
                    reaction_pct = 0
                    if len(price_hist) >= 2:
                        reaction_pct = ((price_hist['Close'].iloc[-1] - price_hist['Close'].iloc[-2]) /
                                        price_hist['Close'].iloc[-2]) * 100

                    overall = "MISS" if (eps_actual or 0) < (eps_est or 0) else "BEAT" if (eps_actual or 0) > (
                                eps_est or 0) else "INLINE"

                    result['earnings_summary'] = {
                        'eps_actual': eps_actual,
                        'eps_estimate': eps_est,
                        'eps_surprise_pct': surprise_pct,
                        'reaction_pct': reaction_pct,
                        'overall_result': overall,
                    }
            except Exception as e:
                logger.debug(f"Earnings result error: {e}")

    except Exception as e:
        logger.error(f"Earnings-aware analysis error for {ticker}: {e}")

    return result'''

new_run_earnings = '''def _run_earnings_aware_analysis(ticker: str, earnings_status: str) -> Dict:
    """Run analysis with earnings context. Uses subprocess for all yfinance calls."""
    from src.analytics.yf_subprocess import get_stock_info, get_earnings_history, get_stock_history

    result = {
        'ticker': ticker,
        'news_count': 0,
        'sentiment_score': None,
        'earnings_summary': None,
    }

    try:
        info = get_stock_info(ticker) or {}
        company_name = info.get('shortName', info.get('longName', ticker))

        # Build earnings-specific search queries
        if earnings_status == 'post':
            queries = [
                f"{ticker} earnings results",
                f"{company_name} quarterly earnings",
                f"{ticker} earnings reaction",
                f"{ticker} guidance outlook",
            ]
        else:
            queries = [
                f"{ticker} earnings preview expectations",
                f"{company_name} earnings whisper",
            ]

        nc = NewsCollector()
        all_articles = []

        # Collect with earnings queries
        for query in queries:
            try:
                articles = nc.collect_ai_search(ticker, company_name=query)
                for a in articles:
                    a['ticker'] = ticker
                all_articles.extend(articles)
            except:
                pass

        # Also standard collection with force refresh
        standard = nc.collect_and_save(ticker, days_back=5, force_refresh=True)
        all_articles.extend(standard.get('articles', []))

        # Deduplicate
        seen = set()
        unique = []
        for a in all_articles:
            title = str(a.get('title', '')).lower()[:40]
            if title and title not in seen:
                seen.add(title)
                unique.append(a)

        result['news_count'] = len(unique)

        # Save and analyze sentiment
        if unique:
            nc.save_articles(unique)

            sa = SentimentAnalyzer()
            sent_result = sa.analyze_ticker_sentiment(ticker, unique)
            if sent_result:
                result['sentiment_score'] = sent_result.get('sentiment_score')

        # Get earnings result if post-earnings
        if earnings_status == 'post':
            try:
                hist = get_earnings_history(ticker)
                if hist is not None and not hist.empty:
                    latest = hist.iloc[0]
                    eps_actual = latest.get('epsActual')
                    eps_est = latest.get('epsEstimate')

                    surprise_pct = None
                    if eps_actual and eps_est and eps_est != 0:
                        surprise_pct = ((eps_actual - eps_est) / abs(eps_est)) * 100

                    # Get price reaction via subprocess
                    price_hist = get_stock_history(ticker, period="5d")
                    reaction_pct = 0
                    if price_hist is not None and len(price_hist) >= 2:
                        reaction_pct = ((price_hist['Close'].iloc[-1] - price_hist['Close'].iloc[-2]) /
                                        price_hist['Close'].iloc[-2]) * 100

                    overall = "MISS" if (eps_actual or 0) < (eps_est or 0) else "BEAT" if (eps_actual or 0) > (
                                eps_est or 0) else "INLINE"

                    result['earnings_summary'] = {
                        'eps_actual': eps_actual,
                        'eps_estimate': eps_est,
                        'eps_surprise_pct': surprise_pct,
                        'reaction_pct': reaction_pct,
                        'overall_result': overall,
                    }
            except Exception as e:
                logger.debug(f"Earnings result error: {e}")

    except Exception as e:
        logger.error(f"Earnings-aware analysis error for {ticker}: {e}")

    return result'''

if old_run_earnings in content:
    content = content.replace(old_run_earnings, new_run_earnings)
    print("PATCH 3: Fixed _run_earnings_aware_analysis() to use subprocess")
else:
    print("PATCH 3: WARNING — Could not find _run_earnings_aware_analysis()")


with open('src/tabs/signals_tab/analysis.py', 'w', encoding='utf-8') as f:
    f.write(content)

# ============================================================================
# VERIFY
# ============================================================================

print("\nVerifying syntax...")
for fp in ['src/analytics/yf_subprocess.py', 'src/tabs/signals_tab/analysis.py']:
    with open(fp, 'r', encoding='utf-8') as f:
        ast.parse(f.read())
    print(f"  {fp}: OK")

# Check remaining direct yfinance calls in analysis.py
print("\nRemaining yfinance references in analysis.py:")
with open('src/tabs/signals_tab/analysis.py', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if 'yf.Ticker' in line or ('import yfinance' in line and 'subprocess' not in line and 'yf_subprocess' not in line):
            stripped = line.strip()
            # Skip lines inside subprocess strings
            if 'stock = yf.Ticker(\\"{' in stripped or 'import json, yfinance' in stripped:
                print(f"  Line {i}: {stripped} [SAFE - inside subprocess string]")
            else:
                print(f"  Line {i}: {stripped} [WARNING - direct call]")

print("\nDone!")