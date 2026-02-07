# ============================================
# PART 1: Create the subprocess worker file
# ============================================
worker_code = '''from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict

import pandas as pd


def _df_to_split_json(df: pd.DataFrame) -> str:
    return df.to_json(orient="split")


def main(argv: list[str]) -> int:
    logging.getLogger().setLevel(logging.CRITICAL)
    try:
        import yfinance as yf

        if len(argv) < 2:
            print(json.dumps({"ok": False, "error": "missing_ticker"}))
            return 2

        ticker = argv[1].strip().upper()
        max_expiries = int(argv[2]) if len(argv) >= 3 else 4
        max_expiries = max(0, max_expiries)

        stock = yf.Ticker(ticker)

        sp = 0.0
        try:
            info = stock.info or {}
            sp = float(
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
                or 0
            )
        except Exception:
            sp = 0.0

        if not sp:
            try:
                hist = stock.history(period="1d")
                if hist is not None and not hist.empty:
                    sp = float(hist["Close"].iloc[-1])
            except Exception:
                pass

        exps = stock.options or []
        if not exps:
            payload: Dict[str, Any] = {
                "ok": True,
                "stock_price": sp,
                "calls_split": None,
                "puts_split": None,
            }
            print(json.dumps(payload))
            return 0

        expiries_to_fetch = list(exps[:max_expiries])
        all_calls = []
        all_puts = []

        for expiry in expiries_to_fetch:
            try:
                opt_chain = stock.option_chain(expiry)
                calls = opt_chain.calls.copy()
                puts = opt_chain.puts.copy()
                calls["expiry"] = expiry
                puts["expiry"] = expiry
                exp_date = datetime.strptime(expiry, "%Y-%m-%d")
                days_to_exp = (exp_date - datetime.now()).days
                calls["daysToExpiry"] = days_to_exp
                puts["daysToExpiry"] = days_to_exp
                all_calls.append(calls)
                all_puts.append(puts)
            except Exception:
                continue

        calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
        puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

        payload = {
            "ok": True,
            "stock_price": sp,
            "calls_split": _df_to_split_json(calls_df),
            "puts_split": _df_to_split_json(puts_df),
        }
        print(json.dumps(payload))
        return 0

    except Exception as e:
        print(json.dumps({"ok": False, "error": repr(e)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
'''

with open('src/analytics/yahoo_options_subprocess.py', 'w', encoding='utf-8') as f:
    f.write(worker_code)
print("PART 1: Created src/analytics/yahoo_options_subprocess.py")

# ============================================
# PART 2: Update options_flow.py
# ============================================
with open('src/analytics/options_flow.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove old multiprocessing helpers if present
if '_yahoo_options_worker' in content:
    # Find and remove the module-level helpers block
    helper_start = content.find('# =============================================================================\n# Yahoo options timeout helpers')
    if helper_start == -1:
        helper_start = content.find('def _yahoo_options_worker')
    
    helper_end = content.find('\nclass OptionsFlowAnalyzer')
    if helper_start != -1 and helper_end != -1:
        content = content[:helper_start] + content[helper_end+1:]
        print("PART 2a: Removed old multiprocessing helpers")

# Remove old import if present
content = content.replace('import multiprocessing as mp\n', '')

# Replace Yahoo fallback section
start_markers = [
    '        # Fallback to Yahoo Finance (HARD timeout via separate process)',
    '        # Fallback to Yahoo Finance (with daemon thread timeout)',
    '        # Fallback to Yahoo Finance (with request-level timeout)',
    '        # Fallback to Yahoo Finance (with socket-level timeout protection)',
    '        # Fallback to Yahoo Finance',
]

start_pos = -1
for marker in start_markers:
    start_pos = content.find(marker)
    if start_pos != -1:
        print(f"PART 2b: Found Yahoo section at marker: {marker[:50]}...")
        break

end_marker = '    def detect_unusual_activity'
end_pos = content.find(end_marker)

if start_pos == -1 or end_pos == -1:
    print(f"ERROR: start={start_pos}, end={end_pos}")
    exit(1)

new_yahoo = '''        # Fallback to Yahoo Finance (HARD timeout via subprocess.run)
        logger.info(f"\\U0001f4ca {ticker}: Using Yahoo Finance options (15-20 min delayed)")

        try:
            import json
            import subprocess
            import sys

            cmd = [
                sys.executable,
                "-m",
                "src.analytics.yahoo_options_subprocess",
                ticker,
                str(int(max_expiries)),
            ]

            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                logger.warning(f"{ticker}: Yahoo options timed out after 15s (subprocess killed)")
                return pd.DataFrame(), pd.DataFrame(), 0.0, "YAHOO"

            stdout = (completed.stdout or "").strip()
            if not stdout:
                err = (completed.stderr or "").strip()
                logger.debug(f"{ticker}: Yahoo subprocess empty output. stderr={err[:2000]}")
                return pd.DataFrame(), pd.DataFrame(), 0.0, "YAHOO"

            lines = [ln for ln in stdout.splitlines() if ln.strip()]
            last = lines[-1] if lines else stdout

            try:
                payload = json.loads(last)
            except Exception:
                logger.debug(f"{ticker}: Yahoo subprocess JSON parse failed. tail={last[-500:]}")
                return pd.DataFrame(), pd.DataFrame(), 0.0, "YAHOO"

            if not payload.get("ok"):
                logger.debug(f"{ticker}: Yahoo subprocess failed: {payload.get('error')}")
                return pd.DataFrame(), pd.DataFrame(), 0.0, "YAHOO"

            stock_price = float(payload.get("stock_price") or 0.0)

            calls_split = payload.get("calls_split")
            puts_split = payload.get("puts_split")

            if not calls_split or not puts_split:
                return pd.DataFrame(), pd.DataFrame(), stock_price, "YAHOO"

            calls_df = pd.read_json(calls_split, orient="split")
            puts_df = pd.read_json(puts_split, orient="split")

            return calls_df, puts_df, stock_price, "YAHOO"

        except Exception as e:
            logger.warning(f"{ticker}: Yahoo fallback failed: {e}")
            return pd.DataFrame(), pd.DataFrame(), 0.0, "YAHOO"

'''

content = content[:start_pos] + new_yahoo + content[end_pos:]
print("PART 2c: Replaced Yahoo fallback with subprocess version")

with open('src/analytics/options_flow.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\nVerifying syntax...")
import ast
ast.parse(content)
print("options_flow.py: SYNTAX OK")

with open('src/analytics/yahoo_options_subprocess.py', 'r', encoding='utf-8') as f:
    ast.parse(f.read())
print("yahoo_options_subprocess.py: SYNTAX OK")
