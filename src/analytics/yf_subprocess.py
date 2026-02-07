"""
Universal yfinance subprocess wrapper.
All yfinance calls go through subprocess to avoid curl_cffi deadlock with Streamlit.

Phase 3 changes:
- Ticker sanitization: all public functions validate tickers against a strict regex
  before embedding them in subprocess scripts (prevents code injection).
- Explicit auto_adjust=True policy on every .history() call, so return calculations
  are correct across stock splits regardless of yfinance version defaults.
- Consistent price column: always uses 'Close' (which equals adjusted close when
  auto_adjust=True). Documented for callers.

Price policy:
  auto_adjust=True means yfinance's 'Close' column IS the split/dividend-adjusted
  close. This is correct for computing forward returns (e.g. return_5d) because if a
  2:1 split happens between entry and exit, the adjusted series reflects that.
  The backfill job records op_price at signal time, then computes returns from
  future adjusted closes — both sides of the ratio are in the same adjusted basis
  as long as you fetch history that covers the entry date.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from typing import Any, Dict, Optional

import pandas as pd
from io import StringIO

logger = logging.getLogger(__name__)

_TIMEOUT = 8  # seconds per subprocess call

# Strict ticker regex: 1-20 chars of letters, digits, dots, hyphens, carets, equals
# Covers: AAPL, BRK.B, BF-B, ^GSPC, 0700.HK, SHOP.TO, EUR=X
_TICKER_RE = re.compile(r'^[A-Za-z0-9.\-^=]{1,20}$')

# Explicit auto_adjust policy (see module docstring for rationale)
_AUTO_ADJUST = True


def _sanitize_ticker(ticker: str) -> str:
    """
    Validate and return a sanitized ticker string.
    Raises ValueError for tickers that don't match the allowed pattern.
    This prevents code injection via f-string subprocess scripts.
    """
    ticker = str(ticker).strip()
    if not _TICKER_RE.match(ticker):
        raise ValueError(f"Invalid ticker format: {ticker!r}")
    return ticker


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
    ticker = _sanitize_ticker(ticker)
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


def get_stock_history(
    ticker: str,
    period: str = "3mo",
    interval: str = None,
    timeout: int = 10,
) -> Optional[pd.DataFrame]:
    """
    Get stock.history() via subprocess. Returns None on failure.

    Always passes auto_adjust={_AUTO_ADJUST} so 'Close' is the adjusted close,
    consistent across yfinance versions and correct for return calculations.
    """
    ticker = _sanitize_ticker(ticker)
    interval_arg = f', interval="{interval}"' if interval else ''
    script = f"""
import json, yfinance as yf
try:
    hist = yf.Ticker("{ticker}").history(period="{period}"{interval_arg}, auto_adjust={_AUTO_ADJUST})
    if hist is not None and not hist.empty:
        print(json.dumps({{"ok": True, "data": hist.to_json(orient="split")}}))
    else:
        print(json.dumps({{"ok": True, "data": None}}))
except Exception as e:
    print(json.dumps({{"ok": False, "error": str(e)}}))
"""
    result = _run_yf_command(script, timeout=timeout)
    if result and result.get("ok") and result.get("data"):
        return pd.read_json(StringIO(result["data"]), orient="split")
    return None


def get_earnings_dates(ticker: str) -> Optional[pd.DataFrame]:
    """Get stock.earnings_dates via subprocess. Returns None on failure."""
    ticker = _sanitize_ticker(ticker)
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
    ticker = _sanitize_ticker(ticker)
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
            hist = stock.history(period="1d", auto_adjust={_AUTO_ADJUST})
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
    ticker = _sanitize_ticker(ticker)
    script = f"""
import json, yfinance as yf
try:
    stock = yf.Ticker("{ticker}")
    info = stock.info or {{}}
    hist = stock.history(period="{history_period}", auto_adjust={_AUTO_ADJUST})
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


def get_earnings_history(ticker: str, timeout: int = 8) -> Optional[pd.DataFrame]:
    """
    Get earnings history (EPS actual vs estimate) via subprocess.
    Returns DataFrame with epsActual, epsEstimate, epsDifference, surprisePercent.
    """
    ticker = _sanitize_ticker(ticker)
    script = f"""
import json, yfinance as yf, traceback
try:
    stock = yf.Ticker("{ticker}")
    eh = stock.earnings_history
    if eh is not None and not eh.empty:
        records = []
        for idx, row in eh.iterrows():
            rec = {{"date": str(idx)[:19]}}
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