"""
Refresh 52-week high/low data for specified tickers.

Usage:
    python refresh_52w_data.py XOM
    python refresh_52w_data.py XOM AAPL MSFT
"""

import sys
import yfinance as yf

def refresh_52w(tickers: list):
    """Fetch and display fresh 52W data from Yahoo Finance."""

    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Refreshing 52W data for {ticker.upper()}")
        print('='*50)

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get fresh data
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            week_52_high = info.get('fiftyTwoWeekHigh', 0)
            week_52_low = info.get('fiftyTwoWeekLow', 0)

            # Calculate percentages
            if week_52_high and current_price:
                pct_from_high = ((current_price - week_52_high) / week_52_high) * 100
            else:
                pct_from_high = None

            if week_52_low and current_price:
                pct_from_low = ((current_price - week_52_low) / week_52_low) * 100
            else:
                pct_from_low = None

            print(f"Current Price: ${current_price:.2f}")
            print(f"52W High:      ${week_52_high:.2f}")
            print(f"52W Low:       ${week_52_low:.2f}")
            print(f"% from High:   {pct_from_high:+.2f}%" if pct_from_high else "% from High: N/A")
            print(f"% from Low:    {pct_from_low:+.2f}%" if pct_from_low else "% from Low: N/A")

            # Check if at/near high
            if pct_from_high is not None and pct_from_high > -1:
                print(f"\n✅ {ticker} is AT or VERY NEAR its 52-week high!")
            elif pct_from_high is not None and pct_from_high > -5:
                print(f"\n📈 {ticker} is NEAR its 52-week high")

            # Check if at/near low
            if pct_from_low is not None and pct_from_low < 5:
                print(f"\n⚠️ {ticker} is AT or VERY NEAR its 52-week low!")
            elif pct_from_low is not None and pct_from_low < 10:
                print(f"\n📉 {ticker} is NEAR its 52-week low")

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python refresh_52w_data.py TICKER [TICKER2 ...]")
        print("Example: python refresh_52w_data.py XOM")
        sys.exit(1)

    tickers = sys.argv[1:]
    refresh_52w(tickers)