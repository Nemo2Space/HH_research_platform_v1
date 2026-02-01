"""
Update Market Cap for All Tickers
==================================
Fetches market cap from yfinance and updates the database.
"""

import sys
sys.path.insert(0, '..')

import yfinance as yf
import pandas as pd
from datetime import datetime, date
import time

# Your biotech tickers
BIOTECH_TICKERS = [
    "ADPT", "DYN", "CERT", "COGT", "SRPT", "MNKD", "CDTX", "BCRX", "ARDX", "TXG",
    "VRDN", "TWST", "EWTX", "AVDL", "JANX", "SDGR", "ABCL", "NVAX", "STOK", "AMLX",
    "NTLA", "PCRX", "XERS", "GPCR", "NEOG", "ELVN", "BBNX", "PGEN", "QURE", "WVE",
    "OPK", "DVAX", "ORIC", "MDXG", "TRVI", "ATAI", "SPRY", "CRMD", "FTRE", "ABUS",
    "SANA", "TSHA", "PHAT", "IOVA", "GERN", "AVXL", "IMNM", "GOSS", "AKBA", "SVRA",
    "PROK", "DAWN", "TNGX", "KURA", "KALV", "VIR", "NRIX", "RLAY", "MRVI", "MYGN",
    "RZLT", "TERN", "CMPX", "AQST", "VSTM", "ATYR", "ESPR", "PRME", "PSNL", "SRDX",
    "XOMA", "REPL", "ERAS", "CRVS", "ATXS", "LXRX", "ALT", "ALDX", "ABEO", "CTMX",
    "OCGN", "LRMR", "RCKT", "PACB", "IMRX", "AUTL", "FULC", "ABSI"
]

def get_db_connection():
    """Get database connection."""
    try:
        from src.db.connection import get_connection
        cm = get_connection()
        return cm.__enter__()
    except (ImportError, AttributeError):
        import psycopg2
        return psycopg2.connect(
            host="localhost", port=5432, dbname="alpha_platform",
            user="alpha", password="alpha_secure_2024"
        )

def fetch_market_cap(ticker: str) -> dict:
    """Fetch market cap and related data from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'price': info.get('currentPrice') or info.get('regularMarketPrice'),
        }
    except Exception as e:
        print(f"   Error fetching {ticker}: {e}")
        return {}

def main():
    print("=" * 70)
    print("UPDATE MARKET CAP FOR BIOTECH TICKERS")
    print("=" * 70)

    conn = get_db_connection()

    # First check table schema
    print("\n1. Checking database schema...")
    with conn.cursor() as cur:
        # Check screener_scores columns
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'screener_scores'
            ORDER BY ordinal_position
        """)
        screener_cols = [r[0] for r in cur.fetchall()]
        print(f"   screener_scores columns: {len(screener_cols)}")
        print(f"   Has market_cap: {'market_cap' in screener_cols}")

        # Check fundamentals columns
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'fundamentals'
            ORDER BY ordinal_position
        """)
        fund_cols = [r[0] for r in cur.fetchall()]
        print(f"   fundamentals columns: {len(fund_cols)}")
        print(f"   Has market_cap: {'market_cap' in fund_cols}")

    # Determine which table to update
    update_screener = 'market_cap' in screener_cols
    update_fundamentals = 'market_cap' in fund_cols

    if not update_screener and not update_fundamentals:
        print("\n   ⚠ Neither table has market_cap column!")
        print("   Adding market_cap column to screener_scores...")
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE screener_scores ADD COLUMN IF NOT EXISTS market_cap NUMERIC")
            conn.commit()
        update_screener = True
        print("   ✓ Column added")

    # Check current state
    print("\n2. Checking current market_cap status...")
    with conn.cursor() as cur:
        if update_fundamentals:
            cur.execute("""
                SELECT ticker, market_cap 
                FROM fundamentals 
                WHERE ticker = ANY(%s)
                AND market_cap IS NOT NULL
                ORDER BY market_cap DESC
                LIMIT 10
            """, (BIOTECH_TICKERS,))
            rows = cur.fetchall()

            if rows:
                print(f"   Found {len(rows)} with market_cap in fundamentals:")
                for ticker, mc in rows[:5]:
                    mc_str = f"${mc/1e9:.2f}B" if mc >= 1e9 else f"${mc/1e6:.0f}M"
                    print(f"     {ticker}: {mc_str}")
            else:
                print("   No market_cap data found in fundamentals")

    # Fetch and update
    print(f"\n3. Fetching market cap from yfinance for {len(BIOTECH_TICKERS)} tickers...")

    updated = 0
    failed = []
    today = date.today()

    for i, ticker in enumerate(BIOTECH_TICKERS):
        print(f"   [{i+1}/{len(BIOTECH_TICKERS)}] {ticker}...", end=" ")

        data = fetch_market_cap(ticker)

        if data.get('market_cap'):
            mc = data['market_cap']
            mc_str = f"${mc/1e9:.2f}B" if mc >= 1e9 else f"${mc/1e6:.0f}M"

            try:
                with conn.cursor() as cur:
                    # Update fundamentals table
                    if update_fundamentals:
                        cur.execute("""
                            UPDATE fundamentals 
                            SET market_cap = %s
                            WHERE ticker = %s
                        """, (mc, ticker))

                    # Update screener_scores if it has the column
                    if update_screener:
                        cur.execute("""
                            UPDATE screener_scores 
                            SET market_cap = %s
                            WHERE ticker = %s
                        """, (mc, ticker))

                    conn.commit()
                    print(f"✓ {mc_str}")
                    updated += 1
            except Exception as e:
                print(f"✗ DB error: {e}")
                conn.rollback()
                failed.append(ticker)
        else:
            print("✗ No data from yfinance")
            failed.append(ticker)

        # Rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    # Summary
    print(f"\n4. Summary:")
    print(f"   Updated: {updated}/{len(BIOTECH_TICKERS)}")
    print(f"   Failed: {len(failed)}")

    if failed:
        print(f"   Failed tickers: {failed}")

    # Verify
    print("\n5. Verification - Top 15 by market cap:")
    with conn.cursor() as cur:
        if update_fundamentals:
            cur.execute("""
                SELECT ticker, market_cap 
                FROM fundamentals 
                WHERE ticker = ANY(%s)
                AND market_cap IS NOT NULL
                ORDER BY market_cap DESC
                LIMIT 15
            """, (BIOTECH_TICKERS,))
        else:
            cur.execute("""
                SELECT ticker, market_cap 
                FROM screener_scores 
                WHERE ticker = ANY(%s)
                AND market_cap IS NOT NULL
                ORDER BY market_cap DESC
                LIMIT 15
            """, (BIOTECH_TICKERS,))

        rows = cur.fetchall()
        if rows:
            for ticker, mc in rows:
                mc_str = f"${mc/1e9:.2f}B" if mc >= 1e9 else f"${mc/1e6:.0f}M"
                print(f"   {ticker:6}: {mc_str}")
        else:
            print("   No data found!")

    conn.close()
    print("\n" + "=" * 70)
    print("DONE - Now run: python debug_portfolio_etf.py")
    print("=" * 70)

if __name__ == "__main__":
    main()