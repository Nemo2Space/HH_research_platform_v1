"""
Update Sector for Existing Tickers
===================================
Quick script to populate sector column in fundamentals table.
"""

import yfinance as yf
from datetime import date
import sys

# Default tickers if none provided
DEFAULT_TICKERS = [
    "ABEO", "CRMD", "NVAX", "CTMX", "MDXG", "XOMA", "STOK", "OCGN", "ERAS", "DAWN",
    "GPCR", "LXRX", "ALT", "CMPX", "ORIC", "PGEN", "NRIX", "BCRX", "CRVS", "TRVI",
    "FTRE", "NEOG", "ARDX", "TERN", "XERS", "ESPR", "LRMR", "EWTX", "ABSI", "KURA",
    "TWST", "REPL", "RCKT", "ATAI", "VIR", "PROK", "ALDX", "KALV", "AMLX", "PSNL",
    "AQST", "MRVI", "ADPT", "JANX", "IMRX", "DYN", "MYGN", "SANA", "AUTL", "SDGR",
    "TXG", "OPK", "PCRX", "FULC", "SPRY", "VRDN", "CDTX", "VSTM", "WVE", "TNGX",
    "ABCL", "PACB", "GOSS", "NTLA", "SVRA", "AKBA", "COGT", "BBNX", "PRME", "QURE",
    "DVAX", "ELVN", "RLAY", "ATYR", "SRDX", "ATXS", "ABUS", "RZLT", "SRPT", "MNKD",
    "CERT", "AVDL"
]


def update_sectors(tickers):
    from src.db.connection import get_connection

    print(f"Updating sector for {len(tickers)} tickers...")

    success = 0
    failed = 0

    for ticker in tickers:
        try:
            # Get sector from yfinance
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            sector = info.get('sector')

            if not sector:
                print(f"  {ticker}: No sector data")
                failed += 1
                continue

            # Update in database
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE fundamentals SET sector = %s WHERE ticker = %s
                    """, (sector, ticker))
                conn.commit()

            print(f"  {ticker}: {sector} ✓")
            success += 1

        except Exception as e:
            print(f"  {ticker}: ERROR - {e}")
            failed += 1

    print(f"\nDone! Success: {success}, Failed: {failed}")


if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS
    update_sectors(tickers)