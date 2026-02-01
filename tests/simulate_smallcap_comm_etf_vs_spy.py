# simulate_smallcap_comm_etf_vs_spy.py
# pip install yfinance pandas matplotlib

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

warnings.filterwarnings("ignore")

TICKERS = [
    "IQ", "IMAX", "DV", "TV", "YELP", "WBTN", "WLY", "STGW", "ZD",
    "LILAK", "LILA", "GLIBA", "GLIBK", "GTN-A", "PLTK", "OPRA",
    "IDT", "TBLA", "CCOI", "MOMO", "SIFY", "SBGI", "CRTO", "CCO", "NMAX"
]

BENCH = "SPY"

@dataclass
class Result:
    prices: pd.DataFrame
    included: list[str]
    missing: list[str]

def download_adj_close(tickers: list[str], start: date, end: date) -> Result:
    # yfinance accepts a space-separated string or list
    data = yf.download(
        tickers=tickers,
        start=str(start),
        end=str(end + timedelta(days=1)),
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True
    )

    # yfinance returns different shapes depending on tickers count
    # We normalize to a DataFrame of Adj Close columns = tickers
    if isinstance(data.columns, pd.MultiIndex):
        if ("Adj Close" in data.columns.get_level_values(0)):
            adj = data["Adj Close"].copy()
        elif ("Close" in data.columns.get_level_values(0)):
            adj = data["Close"].copy()
        else:
            raise RuntimeError("Could not find Adj Close / Close in downloaded data.")
    else:
        # Single ticker case
        if "Adj Close" in data.columns:
            adj = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        elif "Close" in data.columns:
            adj = data[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            raise RuntimeError("Could not find Adj Close / Close in downloaded data.")

    # Drop columns that are entirely NaN (missing tickers / delisted / bad symbols)
    missing = [c for c in adj.columns if adj[c].dropna().empty]
    adj = adj.drop(columns=missing, errors="ignore")

    included = list(adj.columns)

    # Basic cleanup
    adj = adj.sort_index()
    adj = adj.ffill()  # forward-fill within each series
    adj = adj.dropna(how="all")  # drop rows where everything is NaN

    return Result(prices=adj, included=included, missing=missing)

def build_equal_weight_index(adj_close: pd.DataFrame) -> pd.Series:
    # daily returns per ticker
    rets = adj_close.pct_change()

    # equal-weight daily rebalanced portfolio return:
    # average of constituent returns each day (ignoring NaNs)
    port_ret = rets.mean(axis=1, skipna=True)

    # convert to index starting at 100
    idx = (1.0 + port_ret.fillna(0.0)).cumprod() * 100.0
    idx.name = "ETF_EQW"
    return idx

def main():
    end = date.today()
    start = end - timedelta(days=365)

    all_symbols = TICKERS + [BENCH]

    res = download_adj_close(all_symbols, start, end)

    # Separate benchmark
    if BENCH not in res.included:
        raise RuntimeError("SPY data missing; check ticker or data source availability.")

    bench = res.prices[BENCH].rename(BENCH)
    constituents = res.prices.drop(columns=[BENCH], errors="ignore")

    # Exclude constituents with too few observations (e.g., newly listed)
    min_obs = 60
    good_cols = [c for c in constituents.columns if constituents[c].count() >= min_obs]
    dropped_short = sorted(list(set(constituents.columns) - set(good_cols)))
    constituents = constituents[good_cols]

    etf_idx = build_equal_weight_index(constituents)
    spy_idx = (bench / bench.iloc[0]) * 100.0
    spy_idx.name = "SPY"

    combined = pd.concat([etf_idx, spy_idx], axis=1).dropna()

    # Performance summary
    etf_total = combined["ETF_EQW"].iloc[-1] / combined["ETF_EQW"].iloc[0] - 1.0
    spy_total = combined["SPY"].iloc[-1] / combined["SPY"].iloc[0] - 1.0

    print(f"Period: {combined.index.min().date()} -> {combined.index.max().date()}")
    print(f"Included constituents: {len(constituents.columns)}")
    print(f"Missing tickers (no data): {sorted(res.missing)}")
    print(f"Dropped (insufficient history < {min_obs} obs): {dropped_short}")
    print(f"ETF_EQW total return: {etf_total:.2%}")
    print(f"SPY total return:     {spy_total:.2%}")

    # Plot
    plt.figure(figsize=(11, 6))
    plt.plot(combined.index, combined["ETF_EQW"], label="Equal-weight Comm Small-cap (your list)")
    plt.plot(combined.index, combined["SPY"], label="SPY")
    plt.title("Last 12 Months Performance (Indexed to 100)")
    plt.xlabel("Date")
    plt.ylabel("Index Level")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
