from __future__ import annotations

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
