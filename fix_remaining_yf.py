import ast
print("=" * 60)
print("FIXING REMAINING DIRECT YFINANCE CALLS")
print("=" * 60)
content = open('src/analytics/yf_subprocess.py', 'r', encoding='utf-8').read()
changes = 0
old = 'def get_stock_history(ticker: str, period: str = "3mo", timeout: int = 10)'
new = 'def get_stock_history(ticker: str, period: str = "3mo", interval: str = None, timeout: int = 10)'
if old in content:
    content = content.replace(old, new)
    changes += 1
old = '    Returns DataFrame with Open, High, Low, Close, Volume columns.\n    """\n    script = f'
new = '    Returns DataFrame with Open, High, Low, Close, Volume columns.\n    """\n    interval_arg = f\', interval="{interval}"\' if interval else \'\'\n    script = f'
if old in content:
    content = content.replace(old, new)
    changes += 1
old = '    hist = stock.history(period="{period}")'
new = '    hist = stock.history(period="{period}"{interval_arg})'
if old in content:
    content = content.replace(old, new)
    changes += 1
content = content.replace('            rec = {{"date": str(idx)[:10]}}', '            rec = {{"date": str(idx)[:19]}}')
open('src/analytics/yf_subprocess.py', 'w', encoding='utf-8').write(content)
print(f"1. yf_subprocess.py: {changes} changes (interval support)")
content = open('src/tabs/signals_tab/deep_dive.py', 'r', encoding='utf-8').read()
changes = 0
old = '        try:\n            import yfinance as yf\n            from datetime import date\n            stock = yf.Ticker(signal.ticker)\n            ed = stock.earnings_dates'
new = '        try:\n            from src.analytics.yf_subprocess import get_earnings_dates\n            from datetime import date\n            ed = get_earnings_dates(signal.ticker)'
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("2a. deep_dive.py: Fixed earnings_dates check")
else:
    print("2a. deep_dive.py: WARNING - earnings_dates block not found")
old = '            if ei and ei.is_post_earnings:\n                import yfinance as yf\n                stock = yf.Ticker(ticker)\n\n                try:\n                    hist = stock.earnings_history'
new = '            if ei and ei.is_post_earnings:\n                from src.analytics.yf_subprocess import get_earnings_history\n\n                try:\n                    hist = get_earnings_history(ticker)'
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("2b. deep_dive.py: Fixed earnings_history")
else:
    print("2b. deep_dive.py: WARNING - earnings_history block not found")
old = '    try:\n        import yfinance as yf\n        import plotly.graph_objects as go\n        from datetime import datetime, timedelta\n\n        stock = yf.Ticker(ticker)\n        info = stock.info'
new = '    try:\n        from src.analytics.yf_subprocess import get_stock_info, get_stock_history\n        import plotly.graph_objects as go\n        from datetime import datetime, timedelta\n\n        info = get_stock_info(ticker) or {}'
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("2c. deep_dive.py: Fixed _render_chart info fetch")
else:
    print("2c. deep_dive.py: WARNING - _render_chart info block not found")
old = '        # Get data for selected period\n        period_code, interval = period_options[selected_period]\n        data = stock.history(period=period_code, interval=interval)'
new = '        # Get data for selected period via subprocess\n        period_code, interval = period_options[selected_period]\n        data = get_stock_history(ticker, period=period_code, interval=interval, timeout=15)'
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("2d. deep_dive.py: Fixed chart stock.history call")
else:
    print("2d. deep_dive.py: WARNING - chart stock.history not found")
old = '        if data.empty:\n            st.caption("Chart unavailable")\n            return'
new = '        if data is None or data.empty:\n            st.caption("Chart unavailable")\n            return'
if old in content:
    content = content.replace(old, new)
    changes += 1
open('src/tabs/signals_tab/deep_dive.py', 'w', encoding='utf-8').write(content)
print(f"   deep_dive.py: {changes} total changes")
content = open('src/tabs/signals_tab/universe_manager.py', 'r', encoding='utf-8').read()
changes = 0
old = "    # Quick validation - check if ticker exists via yfinance\n    try:\n        yf_ticker = yf.Ticker(ticker)\n        info = yf_ticker.info\n        if not info or info.get('regularMarketPrice') is None:\n            try:\n                fast = yf_ticker.fast_info\n                if not hasattr(fast, 'last_price') or fast.last_price is None:\n                    return {'success': False, 'error': f\"Ticker '{ticker}' not found or has no price data\"}\n            except:\n                return {'success': False, 'error': f\"Ticker '{ticker}' not found or has no price data\"}\n    except Exception as e:\n        return {'success': False, 'error': f\"Could not validate ticker '{ticker}': {str(e)}\"}"
new = "    # Quick validation - check if ticker exists via yfinance subprocess (safe from Streamlit freeze)\n    try:\n        from src.analytics.yf_subprocess import get_stock_info\n        info = get_stock_info(ticker) or {}\n        if not info or (info.get('regularMarketPrice') is None and info.get('currentPrice') is None and info.get('previousClose') is None):\n            return {'success': False, 'error': f\"Ticker '{ticker}' not found or has no price data\"}\n    except Exception as e:\n        return {'success': False, 'error': f\"Could not validate ticker '{ticker}': {str(e)}\"}"
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("3. universe_manager.py: Fixed ticker validation")
else:
    print("3. universe_manager.py: WARNING - validation block not found")
open('src/tabs/signals_tab/universe_manager.py', 'w', encoding='utf-8').write(content)
print(f"   universe_manager.py: {changes} total changes")
content = open('src/tabs/signals_tab/ai_chat.py', 'r', encoding='utf-8').read()
changes = 0
old = "        # Fetch LIVE data if DB is stale or empty\n        if use_live:\n            try:\n                stock = yf.Ticker(ticker)\n                info = stock.info\n                stock_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)\n\n                # Get options expirations\n                expirations = stock.options\n                if not expirations:\n                    return _format_stale_options_context(df, \"No options data available\")\n\n                # Get near-term options (up to 4 expiries)\n                all_calls = []\n                all_puts = []\n                expiries_used = []\n\n                for expiry in expirations[:4]:\n                    try:\n                        opt_chain = stock.option_chain(expiry)\n                        all_calls.append(opt_chain.calls)\n                        all_puts.append(opt_chain.puts)\n                        expiries_used.append(expiry)\n                    except:\n                        continue\n\n                if not all_calls or not all_puts:\n                    return _format_stale_options_context(df, \"Could not fetch options data\")\n\n                calls_df = pd.concat(all_calls, ignore_index=True)\n                puts_df = pd.concat(all_puts, ignore_index=True)"
new = "        # Fetch LIVE data if DB is stale or empty\n        if use_live:\n            try:\n                from src.analytics.yf_subprocess import get_stock_info, get_options_chain\n                info = get_stock_info(ticker) or {}\n                stock_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)\n\n                # Get options via subprocess (safe from Streamlit freeze)\n                chain_result = get_options_chain(ticker, max_expiries=4)\n                if chain_result is None:\n                    return _format_stale_options_context(df, \"No options data available\")\n\n                calls_df, puts_df, chain_price = chain_result\n                if calls_df is None or puts_df is None or calls_df.empty or puts_df.empty:\n                    return _format_stale_options_context(df, \"Could not fetch options data\")\n\n                if not stock_price and chain_price:\n                    stock_price = chain_price\n\n                expiries_used = sorted(calls_df['expiry'].unique().tolist()) if 'expiry' in calls_df.columns else [\"aggregated\"]"
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("4a. ai_chat.py: Fixed options chain fetch")
else:
    print("4a. ai_chat.py: WARNING - options fetch block not found")
old = '                if nearest_expiry:\n                    # Get options for just the nearest expiry\n                    try:\n                        nearest_chain = stock.option_chain(nearest_expiry)\n                        max_pain = _calculate_max_pain_live(nearest_chain.calls, nearest_chain.puts)\n                        max_pain_expiry = nearest_expiry\n                    except:\n                        # Fallback to aggregated if single expiry fails\n                        max_pain = _calculate_max_pain_live(calls_df, puts_df)\n                        max_pain_expiry = f"aggregated ({len(expiries_used)} expiries)"'
new = '                if nearest_expiry:\n                    try:\n                        if \'expiry\' in calls_df.columns:\n                            nearest_calls = calls_df[calls_df[\'expiry\'] == nearest_expiry]\n                            nearest_puts = puts_df[puts_df[\'expiry\'] == nearest_expiry]\n                            if not nearest_calls.empty and not nearest_puts.empty:\n                                max_pain = _calculate_max_pain_live(nearest_calls, nearest_puts)\n                                max_pain_expiry = nearest_expiry\n                            else:\n                                max_pain = _calculate_max_pain_live(calls_df, puts_df)\n                                max_pain_expiry = f"aggregated ({len(expiries_used)} expiries)"\n                        else:\n                            max_pain = _calculate_max_pain_live(calls_df, puts_df)\n                            max_pain_expiry = f"aggregated ({len(expiries_used)} expiries)"\n                    except:\n                        max_pain = _calculate_max_pain_live(calls_df, puts_df)\n                        max_pain_expiry = f"aggregated ({len(expiries_used)} expiries)"'
if old in content:
    content = content.replace(old, new)
    changes += 1
    print("4b. ai_chat.py: Fixed nearest expiry max pain")
else:
    print("4b. ai_chat.py: WARNING - nearest_chain block not found")
open('src/tabs/signals_tab/ai_chat.py', 'w', encoding='utf-8').write(content)
print(f"   ai_chat.py: {changes} total changes")
print("\nVerifying syntax...")
for fp in ['src/analytics/yf_subprocess.py', 'src/tabs/signals_tab/deep_dive.py', 'src/tabs/signals_tab/universe_manager.py', 'src/tabs/signals_tab/ai_chat.py']:
    ast.parse(open(fp, encoding='utf-8').read())
    print(f"  {fp}: OK")
print("\nDone!")
