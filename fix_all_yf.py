# ============================================
# PART 1: Create universal yfinance wrapper
# ============================================
wrapper_code = r'''"""
Universal yfinance subprocess wrapper.
All yfinance calls go through subprocess to avoid curl_cffi deadlock with Streamlit.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from typing import Any, Dict, Optional

import pandas as pd
from io import StringIO

logger = logging.getLogger(__name__)

_TIMEOUT = 8  # seconds per subprocess call


def _run_yf_command(script: str, timeout: float = _TIMEOUT) -> Optional[Dict[str, Any]]:
    """Run a yfinance command in a subprocess and return parsed JSON."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        stdout = (result.stdout or "").strip()
        if not stdout:
            return None
        lines = [ln for ln in stdout.splitlines() if ln.strip()]
        return json.loads(lines[-1]) if lines else None
    except subprocess.TimeoutExpired:
        logger.warning(f"yfinance subprocess timed out after {timeout}s")
        return None
    except Exception as e:
        logger.debug(f"yfinance subprocess error: {e}")
        return None


def get_stock_info(ticker: str) -> Dict[str, Any]:
    """Get stock.info via subprocess. Returns empty dict on failure."""
    script = f"""
import json, yfinance as yf
try:
    info = yf.Ticker("{ticker}").info or {{}}
    print(json.dumps({{"ok": True, "info": info}}, default=str))
except Exception as e:
    print(json.dumps({{"ok": False, "error": str(e)}}))
"""
    result = _run_yf_command(script)
    if result and result.get("ok"):
        return result.get("info", {})
    return {}


def get_stock_history(ticker: str, period: str = "3mo") -> Optional[pd.DataFrame]:
    """Get stock.history() via subprocess. Returns None on failure."""
    script = f"""
import json, yfinance as yf
try:
    hist = yf.Ticker("{ticker}").history(period="{period}")
    if hist is not None and not hist.empty:
        print(json.dumps({{"ok": True, "data": hist.to_json(orient="split")}}))
    else:
        print(json.dumps({{"ok": True, "data": None}}))
except Exception as e:
    print(json.dumps({{"ok": False, "error": str(e)}}))
"""
    result = _run_yf_command(script)
    if result and result.get("ok") and result.get("data"):
        return pd.read_json(StringIO(result["data"]), orient="split")
    return None


def get_earnings_dates(ticker: str) -> Optional[pd.DataFrame]:
    """Get stock.earnings_dates via subprocess. Returns None on failure."""
    script = f"""
import json, yfinance as yf
try:
    ed = yf.Ticker("{ticker}").earnings_dates
    if ed is not None and not ed.empty:
        dates = [str(idx)[:19] for idx in ed.index]
        print(json.dumps({{"ok": True, "dates": dates}}))
    else:
        print(json.dumps({{"ok": True, "dates": []}}))
except Exception as e:
    print(json.dumps({{"ok": False, "error": str(e)}}))
"""
    result = _run_yf_command(script)
    if result and result.get("ok") and result.get("dates"):
        return pd.DataFrame(index=[pd.Timestamp(d) for d in result["dates"]])
    return None


def get_options_chain(ticker: str, max_expiries: int = 4) -> Dict[str, Any]:
    """Get options chain via subprocess. Returns dict with calls_df, puts_df, stock_price."""
    script = f"""
import json, yfinance as yf
from datetime import datetime
import pandas as pd

try:
    stock = yf.Ticker("{ticker}")
    info = stock.info or {{}}
    sp = float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose") or 0)

    if not sp:
        try:
            hist = stock.history(period="1d")
            if hist is not None and not hist.empty:
                sp = float(hist["Close"].iloc[-1])
        except: pass

    exps = stock.options or []
    if not exps:
        print(json.dumps({{"ok": True, "stock_price": sp, "calls": None, "puts": None}}))
    else:
        expiries = list(exps[:{max_expiries}])
        all_calls, all_puts = [], []
        for expiry in expiries:
            try:
                oc = stock.option_chain(expiry)
                c, p = oc.calls.copy(), oc.puts.copy()
                c["expiry"], p["expiry"] = expiry, expiry
                ed = datetime.strptime(expiry, "%Y-%m-%d")
                dte = (ed - datetime.now()).days
                c["daysToExpiry"], p["daysToExpiry"] = dte, dte
                all_calls.append(c)
                all_puts.append(p)
            except: continue
        cd = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
        pd2 = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()
        print(json.dumps({{
            "ok": True, "stock_price": sp,
            "calls": cd.to_json(orient="split"),
            "puts": pd2.to_json(orient="split")
        }}))
except Exception as e:
    print(json.dumps({{"ok": False, "error": str(e)}}))
"""
    result = _run_yf_command(script, timeout=15)
    if result and result.get("ok"):
        stock_price = float(result.get("stock_price") or 0)
        calls_json = result.get("calls")
        puts_json = result.get("puts")
        calls_df = pd.read_json(StringIO(calls_json), orient="split") if calls_json else pd.DataFrame()
        puts_df = pd.read_json(StringIO(puts_json), orient="split") if puts_json else pd.DataFrame()
        return {"calls": calls_df, "puts": puts_df, "stock_price": stock_price}
    return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "stock_price": 0.0}


def get_stock_info_and_history(ticker: str, history_period: str = "3mo") -> Dict[str, Any]:
    """Get both info and history in ONE subprocess call (saves startup time)."""
    script = f"""
import json, yfinance as yf
try:
    stock = yf.Ticker("{ticker}")
    info = stock.info or {{}}
    hist = stock.history(period="{history_period}")
    hist_json = hist.to_json(orient="split") if hist is not None and not hist.empty else None
    print(json.dumps({{"ok": True, "info": info, "history": hist_json}}, default=str))
except Exception as e:
    print(json.dumps({{"ok": False, "error": str(e)}}))
"""
    result = _run_yf_command(script, timeout=10)
    if result and result.get("ok"):
        info = result.get("info", {})
        hist_json = result.get("history")
        hist = pd.read_json(StringIO(hist_json), orient="split") if hist_json else None
        return {"info": info, "history": hist}
    return {"info": {}, "history": None}
'''

with open('src/analytics/yf_subprocess.py', 'w', encoding='utf-8') as f:
    f.write(wrapper_code)
print("PART 1: Created src/analytics/yf_subprocess.py")

# ============================================
# PART 2: Patch short_squeeze.py
# ============================================
with open('src/analytics/short_squeeze.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the direct yfinance calls in analyze_ticker
old_squeeze = """    def analyze_ticker(self, ticker: str) -> ShortSqueezeData:
        \"\"\"Analyze a single ticker for short squeeze potential.\"\"\"
        ticker = ticker.upper()
        data = ShortSqueezeData(ticker=ticker)

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                data.calculation_errors.append("Could not fetch stock info")
                return data"""

new_squeeze = """    def analyze_ticker(self, ticker: str) -> ShortSqueezeData:
        \"\"\"Analyze a single ticker for short squeeze potential.\"\"\"
        ticker = ticker.upper()
        data = ShortSqueezeData(ticker=ticker)

        try:
            # Use subprocess wrapper to avoid curl_cffi/Streamlit deadlock
            from src.analytics.yf_subprocess import get_stock_info_and_history
            _yf_data = get_stock_info_and_history(ticker, history_period="3mo")
            info = _yf_data.get("info", {})
            _hist_from_subprocess = _yf_data.get("history")

            if not info:
                data.calculation_errors.append("Could not fetch stock info")
                return data"""

if old_squeeze in content:
    content = content.replace(old_squeeze, new_squeeze)
    print("PART 2a: Patched analyze_ticker init")
else:
    print("PART 2a: WARNING - Could not find exact match for analyze_ticker init")

# Replace stock.history call with cached version
old_hist = """            try:
                hist = stock.history(period="3mo")"""
new_hist = """            try:
                hist = _hist_from_subprocess"""

if old_hist in content:
    content = content.replace(old_hist, new_hist, 1)
    print("PART 2b: Patched stock.history to use subprocess cache")
else:
    print("PART 2b: WARNING - Could not find stock.history call")

with open('src/analytics/short_squeeze.py', 'w', encoding='utf-8') as f:
    f.write(content)

# ============================================
# PART 3: Patch analysis.py STEP 6 (earnings)
# ============================================
with open('src/tabs/signals_tab/analysis.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_earnings = """            if not skip_earnings_fetch:
                import yfinance as yf
                stock = yf.Ticker(next_ticker)
                ed = stock.earnings_dates"""

new_earnings = """            if not skip_earnings_fetch:
                from src.analytics.yf_subprocess import get_earnings_dates
                ed = get_earnings_dates(next_ticker)"""

if old_earnings in content:
    content = content.replace(old_earnings, new_earnings)
    print("PART 3: Patched earnings calendar to use subprocess")
else:
    print("PART 3: WARNING - Could not find earnings fetch block")

with open('src/tabs/signals_tab/analysis.py', 'w', encoding='utf-8') as f:
    f.write(content)

# ============================================
# Verify all syntax
# ============================================
print("\nVerifying syntax...")
import ast
for f in ['src/analytics/yf_subprocess.py', 'src/analytics/short_squeeze.py', 
          'src/analytics/options_flow.py', 'src/tabs/signals_tab/analysis.py']:
    with open(f, 'r', encoding='utf-8') as fh:
        ast.parse(fh.read())
    print(f"  {f}: OK")

print("\nAll done!")
