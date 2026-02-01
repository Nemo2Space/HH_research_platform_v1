"""
Update prices table with recent data from Yahoo Finance.

Usage:
    python update_prices.py XOM
    python update_prices.py XOM AAPL MSFT
    python update_prices.py --all  # Update all tickers in screener_scores
    python update_prices.py --portfolio  # Update portfolio tickers only
"""

import sys
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_engine():
    """Get database engine."""
    from src.db.connection import get_engine as _get_engine
    return _get_engine()

def update_prices(tickers: list, days: int = 60):
    """Update prices for specified tickers."""

    engine = get_engine()
    total_inserted = 0

    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Updating prices for {ticker.upper()}")
        print('='*50)

        try:
            # Fetch recent prices from Yahoo
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{days}d")

            if hist.empty:
                print(f"  ⚠️ No data returned for {ticker}")
                continue

            # Prepare dataframe
            hist = hist.reset_index()
            hist['ticker'] = ticker.upper()
            hist = hist.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            hist = hist[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]

            # Convert date to date only (remove timezone)
            hist['date'] = pd.to_datetime(hist['date']).dt.date

            # Get existing dates for this ticker
            existing = pd.read_sql(
                f"SELECT date FROM prices WHERE ticker = '{ticker.upper()}'",
                engine
            )
            existing_dates = set(pd.to_datetime(existing['date']).dt.date.tolist())

            # Filter to only new dates
            hist_new = hist[~hist['date'].isin(existing_dates)]

            if hist_new.empty:
                print(f"  ✓ Prices already up to date for {ticker}")
            else:
                # Insert new prices
                hist_new.to_sql('prices', engine, if_exists='append', index=False)
                total_inserted += len(hist_new)
                print(f"  ✅ Inserted {len(hist_new)} new price records")
                print(f"  Date range: {hist_new['date'].min()} to {hist_new['date'].max()}")

            # Show current 52W range
            range_df = pd.read_sql(f"""
                SELECT MIN(close) as low_52w, MAX(close) as high_52w, MAX(date) as last_date
                FROM prices WHERE ticker = '{ticker.upper()}'
                AND date > CURRENT_DATE - INTERVAL '365 days'
            """, engine)

            # Also get live 52W from yfinance for comparison
            info = stock.info
            live_high = info.get('fiftyTwoWeekHigh', 0)
            live_low = info.get('fiftyTwoWeekLow', 0)
            live_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

            print(f"\n  📊 52W Range Comparison:")
            print(f"     DB:   ${range_df['low_52w'].iloc[0]:.2f} - ${range_df['high_52w'].iloc[0]:.2f} (last: {range_df['last_date'].iloc[0]})")
            print(f"     Live: ${live_low:.2f} - ${live_high:.2f} (price: ${live_price:.2f})")

            if live_high > range_df['high_52w'].iloc[0] * 1.01:
                print(f"     ⚠️ Live 52W high is higher than DB - DB may be missing recent data")

        except Exception as e:
            print(f"  ❌ Error updating {ticker}: {e}")

    return total_inserted

def get_all_tickers():
    """Get all unique tickers from screener_scores."""
    engine = get_engine()
    df = pd.read_sql("SELECT DISTINCT ticker FROM screener_scores ORDER BY ticker", engine)
    return df['ticker'].tolist()

def get_portfolio_tickers():
    """Get tickers from portfolio_positions."""
    engine = get_engine()
    try:
        df = pd.read_sql("SELECT DISTINCT ticker FROM portfolio_positions WHERE shares > 0", engine)
        return df['ticker'].tolist()
    except:
        return []

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_prices.py TICKER [TICKER2 ...]")
        print("       python update_prices.py --all       # All screener tickers")
        print("       python update_prices.py --portfolio # Portfolio tickers only")
        sys.exit(1)

    if sys.argv[1] == '--all':
        tickers = get_all_tickers()
        print(f"Updating {len(tickers)} tickers from screener_scores...")
    elif sys.argv[1] == '--portfolio':
        tickers = get_portfolio_tickers()
        if not tickers:
            print("No portfolio positions found")
            sys.exit(1)
        print(f"Updating {len(tickers)} portfolio tickers...")
    else:
        tickers = sys.argv[1:]

    total = update_prices(tickers)
    print(f"\n{'='*50}")
    print(f"✅ Done! Inserted {total} total price records")
    print("🔄 Restart Streamlit to see updated 52W data")