import yfinance as yf
import pandas as pd

stock = yf.Ticker("DGX")
data = stock.history(start="2026-01-01", end="2026-02-02")
print(f"Rows: {len(data)}")
print(f"Index type: {type(data.index[0])}")
print(data.tail())

# Test the conversion
data.index = data.index.tz_localize(None).date
print(f"\nAfter conversion: {type(data.index[0])}")
print(f"Dates: {sorted(data.index)[:5]}")

from datetime import date
score_date = date(2026, 1, 2)
future = [d for d in data.index if d > score_date]
print(f"\nFuture dates after {score_date}: {len(future)}")
print(future[:5] if future else "NONE!")
