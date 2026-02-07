"""Fix all remaining direct yfinance calls across 4 files."""
import ast

print("=" * 60)
print("FIXING REMAINING DIRECT YFINANCE CALLS")
print("=" * 60)

# ============================================================
# 1. yf_subprocess.py - Add interval support to get_stock_history
# ============================================================
content = open('src/analytics/yf_subprocess.py', 'r', encoding='utf-8').read()
changes = 0

# Add interval param
old = 'def get_stock_history(ticker: str, period: str = "3mo", timeout: int = 10)'
new = 'def get_stock_history(ticker: str, period: str = "3mo", interval: str = None, timeout: int = 10)'
if old in content:
    content = content.replace(old, new)
    changes += 1

# Add interval_arg before script
old = '    Returns DataFrame with Open, High, Low, Close, Volume columns.\n    """\n    script = f'
new = '    Returns DataFrame with Open, High, Low, Close, Volume columns.\n    """\n    interval_arg = f\', interval="{interval}"\' if interval else \'\'\n    script = f'
if old in content:
    content = content.replace(old, new)
    changes += 1

# Update history call
old = '    hist = stock.history(period="{period}")'
new = '    hist = stock.history(period="{period}"{interval_arg})'
if old in content:
    content = content.replace(old, new)
    changes += 1

# Capture full datetime for intraday charts
old = '            rec = {{"date": str(idx)[:10]}}'
new = '            rec = {{"date": str(idx)[:19]}}'
# This appears in multiple functions, only replace in get_stock_history
# Actually just replace all - it's fine for daily too
content = content.replace(old, new)

open('src/analytics/yf_subprocess.py', 'w', encoding='utf-8').write(content)
print(f"1. yf_subprocess.py: {changes} changes (interval support)")

# ============================================================
# 2. deep_dive.py - Fix 3 direct yfinance calls
# ============================================================
content = open('src/tabs/signals_tab/deep_dive.py', 'r', encoding='utf-8').read()
changes = 0

# 2a: Fix earnings_dates check
old = """        try:
            import yfinance as yf
            from datetime import date
            stock = yf.Ticker(signal.ticker)
            ed = stock.earnings_dates"""
new = """        try:
            from src.analytics.yf_subprocess import get_earnings_dates
            from datetime import date
            ed = get_earnings_dates(signal.ticker)"""
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("2a. deep_dive.py: Fixed earnings_dates check")
else:
    print("2a. deep_dive.py: WARNING - earnings_dates block not found")

# 2b: Fix earnings_history in _load_additional_data
old = """            if ei and ei.is_post_earnings:
                import yfinance as yf
                stock = yf.Ticker(ticker)

                try:
                    hist = stock.earnings_history"""
new = """            if ei and ei.is_post_earnings:
                from src.analytics.yf_subprocess import get_earnings_history

                try:
                    hist = get_earnings_history(ticker)"""
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("2b. deep_dive.py: Fixed earnings_history")
else:
    print("2b. deep_dive.py: WARNING - earnings_history block not found")

# 2c: Fix _render_chart
old = """    try:
        import yfinance as yf
        import plotly.graph_objects as go
        from datetime import datetime, timedelta

        stock = yf.Ticker(ticker)
        info = stock.info"""
new = """    try:
        from src.analytics.yf_subprocess import get_stock_info, get_stock_history
        import plotly.graph_objects as go
        from datetime import datetime, timedelta

        info = get_stock_info(ticker) or {}"""
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("2c. deep_dive.py: Fixed _render_chart info fetch")
else:
    print("2c. deep_dive.py: WARNING - _render_chart info block not found")

# 2c continued: Fix stock.history call in chart
old = """        # Get data for selected period
        period_code, interval = period_options[selected_period]
        data = stock.history(period=period_code, interval=interval)"""
new = """        # Get data for selected period via subprocess
        period_code, interval = period_options[selected_period]
        data = get_stock_history(ticker, period=period_code, interval=interval, timeout=15)"""
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("2d. deep_dive.py: Fixed chart stock.history call")
else:
    print("2d. deep_dive.py: WARNING - chart stock.history not found")

# 2e: Fix the empty check since get_stock_history can return None
old = """        if data.empty:
            st.caption("Chart unavailable")
            return"""
new = """        if data is None or data.empty:
            st.caption("Chart unavailable")
            return"""
if old in content:
    content = content.replace(old, new)
    changes += 1

open('src/tabs/signals_tab/deep_dive.py', 'w', encoding='utf-8').write(content)
print(f"   deep_dive.py: {changes} total changes")

# ============================================================
# 3. universe_manager.py - Fix ticker validation
# ============================================================
content = open('src/tabs/signals_tab/universe_manager.py', 'r', encoding='utf-8').read()
changes = 0

old = """    # Quick validation - check if ticker exists via yfinance
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        if not info or info.get('regularMarketPrice') is None:
            try:
                fast = yf_ticker.fast_info
                if not hasattr(fast, 'last_price') or fast.last_price is None:
                    return {'success': False, 'error': f"Ticker '{ticker}' not found or has no price data"}
            except:
                return {'success': False, 'error': f"Ticker '{ticker}' not found or has no price data"}
    except Exception as e:
        return {'success': False, 'error': f"Could not validate ticker '{ticker}': {str(e)}"}"""
new = """    # Quick validation - check if ticker exists via yfinance subprocess (safe from Streamlit freeze)
    try:
        from src.analytics.yf_subprocess import get_stock_info
        info = get_stock_info(ticker) or {}
        if not info or (info.get('regularMarketPrice') is None and info.get('currentPrice') is None and info.get('previousClose') is None):
            return {'success': False, 'error': f"Ticker '{ticker}' not found or has no price data"}
    except Exception as e:
        return {'success': False, 'error': f"Could not validate ticker '{ticker}': {str(e)}"}"""
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("3. universe_manager.py: Fixed ticker validation")
else:
    print("3. universe_manager.py: WARNING - validation block not found")

open('src/tabs/signals_tab/universe_manager.py', 'w', encoding='utf-8').write(content)
print(f"   universe_manager.py: {changes} total changes")

# ============================================================
# 4. ai_chat.py - Fix options chain fetch
# ============================================================
content = open('src/tabs/signals_tab/ai_chat.py', 'r', encoding='utf-8').read()
changes = 0

old = """        # Fetch LIVE data if DB is stale or empty
        if use_live:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                stock_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

                # Get options expirations
                expirations = stock.options
                if not expirations:
                    return _format_stale_options_context(df, "No options data available")

                # Get near-term options (up to 4 expiries)
                all_calls = []
                all_puts = []
                expiries_used = []

                for expiry in expirations[:4]:
                    try:
                        opt_chain = stock.option_chain(expiry)
                        all_calls.append(opt_chain.calls)
                        all_puts.append(opt_chain.puts)
                        expiries_used.append(expiry)
                    except:
                        continue

                if not all_calls or not all_puts:
                    return _format_stale_options_context(df, "Could not fetch options data")

                calls_df = pd.concat(all_calls, ignore_index=True)
                puts_df = pd.concat(all_puts, ignore_index=True)"""
new = """        # Fetch LIVE data if DB is stale or empty
        if use_live:
            try:
                from src.analytics.yf_subprocess import get_stock_info, get_options_chain
                info = get_stock_info(ticker) or {}
                stock_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

                # Get options via subprocess (safe from Streamlit freeze)
                chain_result = get_options_chain(ticker, max_expiries=4)
                if chain_result is None:
                    return _format_stale_options_context(df, "No options data available")

                calls_df, puts_df, chain_price = chain_result
                if calls_df is None or puts_df is None or calls_df.empty or puts_df.empty:
                    return _format_stale_options_context(df, "Could not fetch options data")

                if not stock_price and chain_price:
                    stock_price = chain_price

                expiries_used = sorted(calls_df['expiry'].unique().tolist()) if 'expiry' in calls_df.columns else ["aggregated"]"""
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("4a. ai_chat.py: Fixed options chain fetch")
else:
    print("4a. ai_chat.py: WARNING - options fetch block not found")

# Fix nearest_chain re-fetch for max pain
old = """                if nearest_expiry:
                    # Get options for just the nearest expiry
                    try:
                        nearest_chain = stock.option_chain(nearest_expiry)
                        max_pain = _calculate_max_pain_live(nearest_chain.calls, nearest_chain.puts)
                        max_pain_expiry = nearest_expiry
                    except:
                        # Fallback to aggregated if single expiry fails
                        max_pain = _calculate_max_pain_live(calls_df, puts_df)
                        max_pain_expiry = f"aggregated ({len(expiries_used)} expiries)\""""
new = """                if nearest_expiry:
                    # Use nearest expiry data from already-fetched chains
                    try:
                        if 'expiry' in calls_df.columns:
                            nearest_calls = calls_df[calls_df['expiry'] == nearest_expiry]
                            nearest_puts = puts_df[puts_df['expiry'] == nearest_expiry]
                            if not nearest_calls.empty and not nearest_puts.empty:
                                max_pain = _calculate_max_pain_live(nearest_calls, nearest_puts)
                                max_pain_expiry = nearest_expiry
                            else:
                                max_pain = _calculate_max_pain_live(calls_df, puts_df)
                                max_pain_expiry = f"aggregated ({len(expiries_used)} expiries)"
                        else:
                            max_pain = _calculate_max_pain_live(calls_df, puts_df)
                            max_pain_expiry = f"aggregated ({len(expiries_used)} expiries)"
                    except:
                        max_pain = _calculate_max_pain_live(calls_df, puts_df)
                        max_pain_expiry = f"aggregated ({len(expiries_used)} expiries)\""""
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("4b. ai_chat.py: Fixed nearest expiry max pain")
else:
    print("4b. ai_chat.py: WARNING - nearest_chain block not found")

open('src/tabs/signals_tab/ai_chat.py', 'w', encoding='utf-8').write(content)
print(f"   ai_chat.py: {changes} total changes")

# ============================================================
# VERIFY SYNTAX
# ============================================================
print("\n" + "=" * 60)
print("VERIFYING SYNTAX")
print("=" * 60)
files = [
    'src/analytics/yf_subprocess.py',
    'src/tabs/signals_tab/deep_dive.py',
    'src/tabs/signals_tab/universe_manager.py',
    'src/tabs/signals_tab/ai_chat.py',
]
all_ok = True
for fp in files:
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        print(f"  {fp}: OK")
    except SyntaxError as e:
        print(f"  {fp}: SYNTAX ERROR - {e}")
        all_ok = False

# ============================================================
# FINAL SWEEP
# ============================================================
print("\n" + "=" * 60)
print("FINAL yf.Ticker SWEEP (should be empty or subprocess-only)")
print("=" * 60)
scan_files = files + [
    'src/tabs/signals_tab/analysis.py',
    'src/tabs/signals_tab/shared.py',
    'src/analytics/short_squeeze.py',
    'src/analytics/options_flow.py',
]
found_any = False
for fp in scan_files:
    try:
        lines = open(fp, 'r', encoding='utf-8').readlines()
        in_subprocess = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Track if we're inside a subprocess string
            if 'subprocess.run' in stripped or '_ed_cmd' in stripped or 'script = f"""' in stripped:
                in_subprocess = True
            if in_subprocess and ('"""' in stripped and not stripped.startswith('"""') and not 'f"""' in stripped):
                in_subprocess = False
            if '"]' in stripped:
                in_subprocess = False

            if 'yf.Ticker' in stripped:
                if in_subprocess or 'yf.Ticker(\\"{' in stripped or 'yf.Ticker("{' in stripped:
                    status = "SAFE (subprocess)"
                else:
                    status = "!! DIRECT CALL !!"
                    found_any = True
                print(f"  {fp}:{i+1}: {stripped[:80]}  [{status}]")
    except Exception as e:
        print(f"  {fp}: Error reading - {e}")

if not found_any:
    print("  No direct yf.Ticker calls found!")
print("\nDone!")