"""
Refresh options flow data for tickers.

This script:
1. Fetches LIVE options data from Yahoo Finance
2. Calculates P/C ratios and max pain correctly
3. Updates the options_flow_daily table
4. Shows debug info to verify calculations

Usage:
    python refresh_options_flow.py XOM
    python refresh_options_flow.py XOM AAPL MSFT
    python refresh_options_flow.py --debug XOM  # Show detailed debug output
"""

import sys
import yfinance as yf
import pandas as pd
from datetime import datetime, date
from typing import Tuple, Optional

def get_engine():
    """Get database engine."""
    from src.db.connection import get_engine as _get_engine
    return _get_engine()

def calculate_max_pain(calls_df: pd.DataFrame, puts_df: pd.DataFrame, debug: bool = False) -> Tuple[float, dict]:
    """
    Calculate max pain price correctly.
    Max pain = strike where option holders lose the most money.

    Returns:
        Tuple of (max_pain_price, debug_info)
    """
    if calls_df.empty or puts_df.empty:
        return 0, {"error": "Empty dataframes"}

    # Get all unique strikes
    all_strikes = sorted(set(
        list(calls_df['strike'].unique()) +
        list(puts_df['strike'].unique())
    ))

    if not all_strikes:
        return 0, {"error": "No strikes found"}

    min_payout = float('inf')
    max_pain_strike = all_strikes[0]
    payout_by_strike = {}

    for test_strike in all_strikes:
        total_payout = 0

        # ITM Calls: strike < test_strike, payout = (test_strike - strike) * OI
        # These calls are in-the-money if stock settles at test_strike
        for _, row in calls_df.iterrows():
            strike = row['strike']
            oi = row.get('openInterest', 0) or 0
            if strike < test_strike:
                total_payout += oi * (test_strike - strike)

        # ITM Puts: strike > test_strike, payout = (strike - test_strike) * OI
        # These puts are in-the-money if stock settles at test_strike
        for _, row in puts_df.iterrows():
            strike = row['strike']
            oi = row.get('openInterest', 0) or 0
            if strike > test_strike:
                total_payout += oi * (strike - test_strike)

        payout_by_strike[test_strike] = total_payout

        if total_payout < min_payout:
            min_payout = total_payout
            max_pain_strike = test_strike

    debug_info = {
        "strikes_analyzed": len(all_strikes),
        "min_strike": min(all_strikes),
        "max_strike": max(all_strikes),
        "max_pain_strike": max_pain_strike,
        "min_payout_value": min_payout,
    }

    if debug:
        # Find top 5 strikes with lowest payout (max pain)
        sorted_payout = sorted(payout_by_strike.items(), key=lambda x: x[1])[:5]
        debug_info["top_5_lowest_payout"] = sorted_payout

    return max_pain_strike, debug_info


def refresh_options_flow(ticker: str, debug: bool = False) -> dict:
    """
    Fetch live options data and update database.

    Returns:
        Dict with results and debug info
    """
    ticker = ticker.upper()
    result = {
        "ticker": ticker,
        "success": False,
        "error": None,
        "data": {}
    }

    print(f"\n{'='*60}")
    print(f"Refreshing options flow for {ticker}")
    print('='*60)

    try:
        # Fetch from Yahoo Finance
        stock = yf.Ticker(ticker)
        info = stock.info
        stock_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

        print(f"Current stock price: ${stock_price:.2f}")

        # Get expiration dates
        try:
            expirations = stock.options
        except Exception as e:
            result["error"] = f"Could not get options expirations: {e}"
            print(f"❌ {result['error']}")
            return result

        if not expirations:
            result["error"] = "No options expirations available"
            print(f"❌ {result['error']}")
            return result

        print(f"Available expirations: {expirations[:5]}{'...' if len(expirations) > 5 else ''}")

        # Get options for near-term expiries (up to 4)
        all_calls = []
        all_puts = []
        expiries_used = []

        for expiry in expirations[:4]:
            try:
                opt_chain = stock.option_chain(expiry)
                calls = opt_chain.calls.copy()
                puts = opt_chain.puts.copy()

                calls['expiry'] = expiry
                puts['expiry'] = expiry

                all_calls.append(calls)
                all_puts.append(puts)
                expiries_used.append(expiry)

                if debug:
                    print(f"  Expiry {expiry}: {len(calls)} calls, {len(puts)} puts")

            except Exception as e:
                print(f"  ⚠️ Error fetching {expiry}: {e}")

        if not all_calls or not all_puts:
            result["error"] = "No options data retrieved"
            print(f"❌ {result['error']}")
            return result

        calls_df = pd.concat(all_calls, ignore_index=True)
        puts_df = pd.concat(all_puts, ignore_index=True)

        print(f"Total options: {len(calls_df)} calls, {len(puts_df)} puts")
        print(f"Expiries used: {expiries_used}")

        # Calculate volumes
        call_volume = calls_df['volume'].sum() if 'volume' in calls_df else 0
        put_volume = puts_df['volume'].sum() if 'volume' in puts_df else 0
        call_oi = calls_df['openInterest'].sum() if 'openInterest' in calls_df else 0
        put_oi = puts_df['openInterest'].sum() if 'openInterest' in puts_df else 0

        # Calculate ratios
        pc_volume_ratio = put_volume / call_volume if call_volume > 0 else 1.0
        pc_oi_ratio = put_oi / call_oi if call_oi > 0 else 1.0

        print(f"\n📊 Volume Analysis:")
        print(f"  Call Volume: {call_volume:,.0f}")
        print(f"  Put Volume: {put_volume:,.0f}")
        print(f"  P/C Ratio (volume): {pc_volume_ratio:.2f}")
        print(f"  P/C Ratio (OI): {pc_oi_ratio:.2f}")

        # Calculate max pain
        max_pain, pain_debug = calculate_max_pain(calls_df, puts_df, debug)

        print(f"\n🎯 Max Pain Analysis:")
        print(f"  Max Pain Strike: ${max_pain:.2f}")
        print(f"  Stock Price: ${stock_price:.2f}")
        print(f"  Difference: {((max_pain - stock_price) / stock_price * 100):+.1f}%")

        if debug and "top_5_lowest_pain" in pain_debug:
            print(f"  Top 5 lowest pain strikes:")
            for strike, pain in pain_debug["top_5_lowest_pain"]:
                print(f"    ${strike:.2f}: {pain:,.0f}")

        # Validate max pain
        if stock_price > 0:
            max_pain_pct_diff = abs(max_pain - stock_price) / stock_price * 100
            if max_pain_pct_diff > 20:
                print(f"  ⚠️ WARNING: Max pain is {max_pain_pct_diff:.0f}% from stock price - unusual but may be valid")

        # Calculate IV
        avg_call_iv = calls_df['impliedVolatility'].mean() if 'impliedVolatility' in calls_df else 0
        avg_put_iv = puts_df['impliedVolatility'].mean() if 'impliedVolatility' in puts_df else 0
        iv_skew = avg_put_iv - avg_call_iv

        # Determine sentiment
        if pc_volume_ratio < 0.5:
            sentiment = 'BULLISH'
            sentiment_score = min(100, 50 + (0.5 - pc_volume_ratio) * 100)
        elif pc_volume_ratio > 1.5:
            sentiment = 'BEARISH'
            sentiment_score = max(-100, -50 - (pc_volume_ratio - 1.5) * 50)
        else:
            sentiment = 'NEUTRAL'
            sentiment_score = 50

        print(f"\n📈 Sentiment: {sentiment} (score: {sentiment_score:.0f})")

        # Prepare data for DB - convert numpy types to native Python
        def to_native(val):
            """Convert numpy types to native Python types."""
            if val is None:
                return None
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            return val

        data = {
            "stock_price": to_native(stock_price),
            "call_volume": int(to_native(call_volume)),
            "put_volume": int(to_native(put_volume)),
            "call_oi": int(to_native(call_oi)),
            "put_oi": int(to_native(put_oi)),
            "pc_volume_ratio": round(float(to_native(pc_volume_ratio)), 4),
            "pc_oi_ratio": round(float(to_native(pc_oi_ratio)), 4),
            "max_pain": round(float(to_native(max_pain)), 2),
            "avg_call_iv": round(float(to_native(avg_call_iv)), 4),
            "avg_put_iv": round(float(to_native(avg_put_iv)), 4),
            "iv_skew": round(float(to_native(iv_skew)), 4),
            "sentiment": sentiment,
            "sentiment_score": round(float(to_native(sentiment_score)), 1),
            "expiries_used": expiries_used,
            "scan_date": date.today().isoformat(),
        }

        result["data"] = data

        # Update database
        try:
            engine = get_engine()
            from sqlalchemy import text

            with engine.connect() as conn:
                # Check if record exists
                check = conn.execute(text(
                    "SELECT id FROM options_flow_daily WHERE ticker = :ticker AND scan_date = :scan_date"
                ), {"ticker": ticker, "scan_date": date.today()})
                exists = check.fetchone()

                if exists:
                    # Update - use data dict which has native types
                    conn.execute(text("""
                        UPDATE options_flow_daily SET
                            stock_price = :stock_price,
                            total_call_volume = :call_volume,
                            total_put_volume = :put_volume,
                            total_call_oi = :call_oi,
                            total_put_oi = :put_oi,
                            put_call_volume_ratio = :pc_volume_ratio,
                            put_call_oi_ratio = :pc_oi_ratio,
                            avg_call_iv = :avg_call_iv,
                            avg_put_iv = :avg_put_iv,
                            iv_skew = :iv_skew,
                            overall_sentiment = :sentiment,
                            sentiment_score = :sentiment_score,
                            max_pain_price = :max_pain
                        WHERE ticker = :ticker AND scan_date = :scan_date
                    """), {
                        "ticker": ticker,
                        "scan_date": date.today(),
                        "stock_price": data["stock_price"],
                        "call_volume": data["call_volume"],
                        "put_volume": data["put_volume"],
                        "call_oi": data["call_oi"],
                        "put_oi": data["put_oi"],
                        "pc_volume_ratio": data["pc_volume_ratio"],
                        "pc_oi_ratio": data["pc_oi_ratio"],
                        "avg_call_iv": data["avg_call_iv"],
                        "avg_put_iv": data["avg_put_iv"],
                        "iv_skew": data["iv_skew"],
                        "sentiment": data["sentiment"],
                        "sentiment_score": data["sentiment_score"],
                        "max_pain": data["max_pain"],
                    })
                    print(f"\n✅ Updated existing record for {ticker}")
                else:
                    # Insert - use data dict which has native types
                    conn.execute(text("""
                        INSERT INTO options_flow_daily 
                            (ticker, scan_date, stock_price, total_call_volume, total_put_volume,
                             total_call_oi, total_put_oi, put_call_volume_ratio, put_call_oi_ratio,
                             avg_call_iv, avg_put_iv, iv_skew, overall_sentiment, sentiment_score,
                             max_pain_price, high_alerts, medium_alerts, low_alerts)
                        VALUES 
                            (:ticker, :scan_date, :stock_price, :call_volume, :put_volume,
                             :call_oi, :put_oi, :pc_volume_ratio, :pc_oi_ratio,
                             :avg_call_iv, :avg_put_iv, :iv_skew, :sentiment, :sentiment_score,
                             :max_pain, 0, 0, 0)
                    """), {
                        "ticker": ticker,
                        "scan_date": date.today(),
                        "stock_price": data["stock_price"],
                        "call_volume": data["call_volume"],
                        "put_volume": data["put_volume"],
                        "call_oi": data["call_oi"],
                        "put_oi": data["put_oi"],
                        "pc_volume_ratio": data["pc_volume_ratio"],
                        "pc_oi_ratio": data["pc_oi_ratio"],
                        "avg_call_iv": data["avg_call_iv"],
                        "avg_put_iv": data["avg_put_iv"],
                        "iv_skew": data["iv_skew"],
                        "sentiment": data["sentiment"],
                        "sentiment_score": data["sentiment_score"],
                        "max_pain": data["max_pain"],
                    })
                    print(f"\n✅ Inserted new record for {ticker}")

                conn.commit()

            result["success"] = True

        except Exception as e:
            result["error"] = f"Database error: {e}"
            print(f"\n❌ Database error: {e}")

    except Exception as e:
        result["error"] = str(e)
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    return result


def verify_db_data(ticker: str):
    """Verify what's in the database after update."""
    print(f"\n📋 Verifying DB data for {ticker}:")

    try:
        engine = get_engine()
        df = pd.read_sql(f"""
            SELECT scan_date, stock_price, total_call_volume, total_put_volume,
                   put_call_volume_ratio, put_call_oi_ratio, max_pain_price,
                   overall_sentiment, sentiment_score
            FROM options_flow_daily 
            WHERE ticker = '{ticker}'
            ORDER BY scan_date DESC
            LIMIT 3
        """, engine)

        if df.empty:
            print("  No data found")
        else:
            print(df.to_string())
    except Exception as e:
        print(f"  Error: {e}")


def get_all_tickers():
    """Get all unique tickers from screener_scores."""
    engine = get_engine()
    df = pd.read_sql("SELECT DISTINCT ticker FROM screener_scores ORDER BY ticker", engine)
    return df['ticker'].tolist()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python refresh_options_flow.py TICKER [TICKER2 ...]")
        print("       python refresh_options_flow.py --all       # All screener tickers")
        print("       python refresh_options_flow.py --debug TICKER")
        sys.exit(1)

    debug = False
    tickers = sys.argv[1:]

    if tickers[0] == '--debug':
        debug = True
        tickers = tickers[1:]

    if tickers[0] == '--all':
        tickers = get_all_tickers()
        print(f"🔄 Refreshing options flow for {len(tickers)} tickers...")
        print("⚠️ This may take 10-20 minutes due to Yahoo rate limits\n")

    success_count = 0
    fail_count = 0

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}]", end="")
        result = refresh_options_flow(ticker, debug)
        if result["success"]:
            success_count += 1
        else:
            fail_count += 1

        # Rate limit - pause between requests to avoid Yahoo blocking
        if len(tickers) > 5:
            import time
            time.sleep(1)  # 1 second delay between tickers

    print(f"\n{'='*60}")
    print(f"✅ Done! Success: {success_count}, Failed: {fail_count}")
    print("🔄 Restart Streamlit to see updated data.")