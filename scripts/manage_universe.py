"""
Alpha Platform - Stock Universe Manager

Single command to add or remove stocks from your universe.
Automatically updates all data (fundamentals, sectors, dates, news, etc.)

Usage:
    # Add a stock
    python scripts/manage_universe.py --add INTC
    python scripts/manage_universe.py --add INTC IBM QCOM

    # Remove a stock
    python scripts/manage_universe.py --remove COIN
    python scripts/manage_universe.py --remove COIN SQ

    # List current universe
    python scripts/manage_universe.py --list

    # Refresh all data for existing stocks
    python scripts/manage_universe.py --refresh
"""

import os
import sys
import argparse
import subprocess
import time

# Setup path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import yfinance as yf

from dotenv import load_dotenv
load_dotenv()

from src.db.connection import get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)

UNIVERSE_FILE = os.path.join(project_root, 'config', 'universe.csv')


def get_stock_info(ticker: str) -> dict:
    """Fetch stock info from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            'ticker': ticker.upper(),
            'name': info.get('longName') or info.get('shortName') or ticker,
            'sector': info.get('sector') or 'Unknown',
            'industry': info.get('industry') or 'Unknown'
        }
    except Exception as e:
        logger.warning(f"Could not fetch info for {ticker}: {e}")
        return {
            'ticker': ticker.upper(),
            'name': ticker.upper(),
            'sector': 'Unknown',
            'industry': 'Unknown'
        }


def load_universe() -> pd.DataFrame:
    """Load current universe from CSV."""
    if os.path.exists(UNIVERSE_FILE):
        return pd.read_csv(UNIVERSE_FILE)
    return pd.DataFrame(columns=['ticker', 'name', 'sector', 'industry'])


def save_universe(df: pd.DataFrame):
    """Save universe to CSV."""
    df.to_csv(UNIVERSE_FILE, index=False)
    logger.info(f"Saved {len(df)} tickers to {UNIVERSE_FILE}")


def add_ticker(ticker: str) -> bool:
    """Add a single ticker to universe."""
    ticker = ticker.upper().strip()

    # Load current universe
    df = load_universe()

    # Check if already exists
    if ticker in df['ticker'].values:
        print(f"⚠️  {ticker} already in universe")
        return False

    # Fetch stock info
    print(f"📡 Fetching info for {ticker}...")
    info = get_stock_info(ticker)

    # Add to dataframe
    new_row = pd.DataFrame([info])
    df = pd.concat([df, new_row], ignore_index=True)

    # Save
    save_universe(df)
    print(f"✅ Added {ticker}: {info['name']} ({info['sector']})")

    return True


def remove_ticker(ticker: str) -> bool:
    """Remove a ticker from universe."""
    ticker = ticker.upper().strip()

    # Load current universe
    df = load_universe()

    # Check if exists
    if ticker not in df['ticker'].values:
        print(f"⚠️  {ticker} not in universe")
        return False

    # Remove from dataframe
    df = df[df['ticker'] != ticker]

    # Save
    save_universe(df)
    print(f"🗑️  Removed {ticker} from universe")

    # Optionally clean from database
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Remove from screener_scores (latest only)
                cur.execute("DELETE FROM screener_scores WHERE ticker = %s", (ticker,))
                # Remove from trading_signals
                cur.execute("DELETE FROM trading_signals WHERE ticker = %s", (ticker,))
                conn.commit()
        print(f"🧹 Cleaned {ticker} data from database")
    except Exception as e:
        logger.warning(f"Could not clean database: {e}")

    return True


def populate_ticker_data(ticker: str):
    """Populate all data for a single ticker."""
    ticker = ticker.upper()

    print(f"\n{'='*50}")
    print(f"📊 Populating data for {ticker}")
    print('='*50)

    # 1. Fundamentals & Sector
    print("\n1️⃣  Fetching fundamentals & sector...")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if fundamentals row exists
                cur.execute("SELECT id FROM fundamentals WHERE ticker = %s", (ticker,))
                exists = cur.fetchone()

                if exists:
                    # Update existing
                    cur.execute("""
                        UPDATE fundamentals SET
                            sector = %s,
                            market_cap = %s,
                            pe_ratio = %s,
                            forward_pe = %s,
                            pb_ratio = %s,
                            roe = %s,
                            profit_margin = %s,
                            revenue_growth = %s,
                            dividend_yield = %s
                        WHERE ticker = %s
                    """, (
                        info.get('sector'),
                        info.get('marketCap'),
                        info.get('trailingPE'),
                        info.get('forwardPE'),
                        info.get('priceToBook'),
                        info.get('returnOnEquity'),
                        info.get('profitMargins'),
                        info.get('revenueGrowth'),
                        info.get('dividendYield'),
                        ticker
                    ))
                else:
                    # Insert new
                    cur.execute("""
                        INSERT INTO fundamentals (ticker, date, sector, market_cap, pe_ratio,
                                                  forward_pe, pb_ratio, roe, profit_margin, revenue_growth, dividend_yield)
                        VALUES (%s, CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                                    ticker,
                                    info.get('sector'),
                                    info.get('marketCap'),
                                    info.get('trailingPE'),
                                    info.get('forwardPE'),
                                    info.get('priceToBook'),
                                    info.get('returnOnEquity'),
                                    info.get('profitMargins'),
                                    info.get('revenueGrowth'),
                                    info.get('dividendYield')
                    ))

                conn.commit()
        print(f"   ✅ Fundamentals saved")
    except Exception as e:
        print(f"   ❌ Fundamentals error: {e}")

    # 2. Earnings & Ex-Dividend dates
    print("\n2️⃣  Fetching earnings & ex-dividend dates...")
    try:
        from scripts.populate_dates import update_ticker_dates
        if update_ticker_dates(ticker):
            print(f"   ✅ Dates updated")
        else:
            print(f"   ⚠️  No dates found")
    except Exception as e:
        print(f"   ❌ Dates error: {e}")

    # 3. Analyst ratings
    print("\n3️⃣  Fetching analyst ratings...")
    try:
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations

        if recommendations is not None and not recommendations.empty:
            # Get latest recommendation summary
            latest = recommendations.iloc[-1] if len(recommendations) > 0 else None

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO analyst_ratings (ticker, date, analyst_total, consensus_rating, source)
                        VALUES (%s, CURRENT_DATE, %s, %s, 'yfinance')
                        ON CONFLICT (ticker, date, source) DO UPDATE SET
                            analyst_total = EXCLUDED.analyst_total,
                            consensus_rating = EXCLUDED.consensus_rating
                    """, (ticker, len(recommendations), 'Mixed'))
                    conn.commit()
            print(f"   ✅ Analyst ratings saved")
        else:
            print(f"   ⚠️  No analyst data")
    except Exception as e:
        print(f"   ❌ Analyst error: {e}")

    # 4. Price data
    print("\n4️⃣  Fetching price data...")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='5d')

        if not hist.empty:
            latest = hist.iloc[-1]

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO prices (ticker, date, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, date) DO UPDATE SET
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume
                    """, (
                                    ticker,
                                    hist.index[-1].date(),
                                    float(latest['Open']),
                                    float(latest['High']),
                                    float(latest['Low']),
                                    float(latest['Close']),
                                    int(latest['Volume'])
                    ))
                    conn.commit()
            print(f"   ✅ Price saved: ${latest['Close']:.2f}")
        else:
            print(f"   ⚠️  No price data")
    except Exception as e:
        print(f"   ❌ Price error: {e}")

    # 5. Price targets
    print("\n5️⃣  Fetching price targets...")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        target_high = info.get('targetHighPrice')
        target_low = info.get('targetLowPrice')
        target_mean = info.get('targetMeanPrice')
        current = info.get('currentPrice') or info.get('regularMarketPrice')

        if target_mean and current:
            upside = ((target_mean - current) / current) * 100

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO price_targets (ticker, date, current_price, target_high, 
                            target_low, target_mean, target_upside_pct)
                        VALUES (%s, CURRENT_DATE, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, date) DO UPDATE SET
                            current_price = EXCLUDED.current_price,
                            target_mean = EXCLUDED.target_mean,
                            target_upside_pct = EXCLUDED.target_upside_pct
                                """, (ticker, current, target_high, target_low, target_mean, upside))
                    conn.commit()
            print(f"   ✅ Targets saved: ${target_mean:.2f} ({upside:+.1f}%)")
        else:
            print(f"   ⚠️  No target data")
    except Exception as e:
        print(f"   ❌ Targets error: {e}")

    print(f"\n{'='*50}")
    print(f"✅ {ticker} data population complete!")
    print('='*50)


def list_universe():
    """List current universe."""
    df = load_universe()

    print("\n" + "=" * 60)
    print(f"CURRENT UNIVERSE ({len(df)} stocks)")
    print("=" * 60)

    # Group by sector
    for sector in sorted(df['sector'].unique()):
        sector_df = df[df['sector'] == sector]
        print(f"\n📁 {sector} ({len(sector_df)})")
        for _, row in sector_df.iterrows():
            print(f"   {row['ticker']:<8} {row['name'][:40]}")

    print("\n" + "=" * 60)


def refresh_all():
    """Refresh data for all tickers in universe."""
    df = load_universe()

    print(f"\n🔄 Refreshing data for {len(df)} stocks...")
    print("This may take a few minutes.\n")

    for i, ticker in enumerate(df['ticker']):
        print(f"\n[{i+1}/{len(df)}] Processing {ticker}...")
        populate_ticker_data(ticker)
        time.sleep(0.5)  # Rate limiting

    print("\n" + "=" * 60)
    print("✅ All stocks refreshed!")
    print("=" * 60)


def run_screener_for_ticker(ticker: str):
    """Run the screener for a single ticker only."""
    print(f"\n📊 Running screener for {ticker}...")
    try:
        screener_path = os.path.join(project_root, "scripts", "run_full_screener.py")
        subprocess.run(
            [sys.executable, screener_path, "--ticker", ticker],
            cwd=project_root
        )
        print(f"✅ Screener complete for {ticker}")
    except Exception as e:
        print(f"❌ Screener error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Manage stock universe")
    parser.add_argument("--add", nargs='+', help="Add ticker(s) to universe")
    parser.add_argument("--remove", nargs='+', help="Remove ticker(s) from universe")
    parser.add_argument("--list", action="store_true", help="List current universe")
    parser.add_argument("--refresh", action="store_true", help="Refresh all data")
    parser.add_argument("--no-data", action="store_true", help="Skip data population (add/remove only)")
    parser.add_argument("--no-screener", action="store_true", help="Skip running screener")

    args = parser.parse_args()

    if args.list:
        list_universe()

    elif args.add:
        added = []
        for ticker in args.add:
            if add_ticker(ticker):
                added.append(ticker.upper())

        # Populate data for added tickers
        if added and not args.no_data:
            print("\n" + "=" * 60)
            print("📊 Populating data for new stocks...")
            print("=" * 60)
            for ticker in added:
                populate_ticker_data(ticker)
                time.sleep(0.5)

                # Run screener ONLY for this new ticker
                if not args.no_screener:
                    run_screener_for_ticker(ticker)

            print("\n" + "=" * 60)
            print("✅ All done! Refresh dashboard to see new stock(s).")
            print("=" * 60)

    elif args.remove:
        for ticker in args.remove:
            remove_ticker(ticker)
        print("\n✅ Done! Refresh dashboard to see changes.")

    elif args.refresh:
        refresh_all()

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/manage_universe.py --add INTC")
        print("  python scripts/manage_universe.py --add INTC IBM QCOM")
        print("  python scripts/manage_universe.py --remove COIN")
        print("  python scripts/manage_universe.py --list")
        print("  python scripts/manage_universe.py --refresh")


if __name__ == "__main__":
    main()
