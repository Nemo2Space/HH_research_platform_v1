"""
Diagnostic script to check why MU earnings shows MISS instead of BEAT
Run: python scripts/diagnose_mu_earnings.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

import yfinance as yf
import pandas as pd

ticker = "MU"
print(f"\n{'='*70}")
print(f"DIAGNOSING EARNINGS DATA FOR {ticker}")
print(f"{'='*70}")

stock = yf.Ticker(ticker)

# 1. Check earnings_dates
print("\n1. EARNINGS_DATES TABLE:")
print("-" * 50)
ed = stock.earnings_dates
if ed is not None and not ed.empty:
    print(f"Columns: {ed.columns.tolist()}")
    print("\nRecent entries:")
    print(ed.head(5).to_string())

    # Get the most recent reported earnings
    from datetime import date
    for idx in ed.index:
        d = idx.date() if hasattr(idx, 'date') else idx
        if d <= date.today():
            row = ed.loc[idx]
            print(f"\n>>> MOST RECENT EARNINGS ({d}):")
            print(f"    Reported EPS: {row.get('Reported EPS')}")
            print(f"    EPS Estimate: {row.get('EPS Estimate')}")
            print(f"    Surprise(%): {row.get('Surprise(%)')}")

            eps_actual = row.get('Reported EPS')
            eps_est = row.get('EPS Estimate')
            surprise = row.get('Surprise(%)')

            if pd.notna(surprise):
                result = "BEAT" if float(surprise) > 0 else "MISS"
                print(f"\n    ==> From Surprise(%): {result}")
            elif pd.notna(eps_actual) and pd.notna(eps_est):
                result = "BEAT" if float(eps_actual) > float(eps_est) else "MISS"
                print(f"\n    ==> Calculated: {result} (actual {eps_actual} vs est {eps_est})")
            break
else:
    print("No earnings_dates data!")

# 2. Check earnings_history
print("\n\n2. EARNINGS_HISTORY TABLE:")
print("-" * 50)
try:
    hist = stock.earnings_history
    if hist is not None and not hist.empty:
        print(f"Columns: {hist.columns.tolist()}")
        print("\nRecent entries:")
        print(hist.head(3).to_string())
    else:
        print("No earnings_history data!")
except Exception as e:
    print(f"Error: {e}")

# 3. Check what ECS calculator returns (if available)
print("\n\n3. ECS CALCULATOR (if available):")
print("-" * 50)
try:
    from src.analytics.earnings_intelligence.ecs_calculator import calculate_ecs
    ecs_result = calculate_ecs(ticker)
    print(f"ECS Category: {ecs_result.ecs_category}")
    print(f"EPS Surprise %: {ecs_result.eps_surprise_pct}")
    print(f"Revenue Surprise %: {ecs_result.revenue_surprise_pct}")
    if hasattr(ecs_result, 'eps_actual'):
        print(f"EPS Actual: {ecs_result.eps_actual}")
    if hasattr(ecs_result, 'eps_estimate'):
        print(f"EPS Estimate: {ecs_result.eps_estimate}")
except ImportError:
    print("ECS calculator not available")
except Exception as e:
    print(f"ECS Error: {e}")

# 4. Check database for stored earnings data
print("\n\n4. DATABASE EARNINGS DATA:")
print("-" * 50)
try:
    from src.db.connection import get_engine
    engine = get_engine()

    # Check earnings_calendar
    df = pd.read_sql(f"SELECT * FROM earnings_calendar WHERE ticker = '{ticker}' ORDER BY earnings_date DESC LIMIT 3", engine)
    if not df.empty:
        print("earnings_calendar:")
        print(df.to_string())
    else:
        print("No earnings_calendar data")

    # Check if there's an earnings_analysis table
    try:
        df2 = pd.read_sql(f"SELECT * FROM earnings_analysis WHERE ticker = '{ticker}' ORDER BY created_at DESC LIMIT 3", engine)
        if not df2.empty:
            print("\nearnings_analysis:")
            print(df2.to_string())
    except:
        pass

except Exception as e:
    print(f"Database error: {e}")

# 5. Check price reaction
print("\n\n5. PRICE REACTION (last 5 days):")
print("-" * 50)
hist = stock.history(period="5d")
if not hist.empty:
    print(hist[['Open', 'High', 'Low', 'Close', 'Volume']].to_string())

    if len(hist) >= 2:
        change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
        print(f"\nLast day change: {change:+.2f}%")

print(f"\n{'='*70}")
print("DIAGNOSIS COMPLETE")
print(f"{'='*70}\n")