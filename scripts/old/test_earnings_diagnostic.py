#!/usr/bin/env python3
"""
Diagnostic script to test Earnings Intelligence for a specific ticker.
Run this to see what data is being retrieved and why.

Usage:
    python scripts/test_earnings_diagnostic.py MU
    python scripts/test_earnings_diagnostic.py AAPL
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def diagnose_ticker(ticker: str):
    """Run full diagnostic on a ticker's earnings data."""

    print("=" * 60)
    print(f"EARNINGS INTELLIGENCE DIAGNOSTIC: {ticker}")
    print("=" * 60)

    # Step 1: Check yfinance raw data
    print("\n1. YFINANCE RAW DATA")
    print("-" * 40)

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)

        # Calendar
        print("Calendar:")
        try:
            cal = stock.calendar
            if cal is not None:
                print(f"  {cal}")
            else:
                print("  None returned")
        except Exception as e:
            print(f"  Error: {e}")

        # Earnings dates
        print("\nEarnings Dates (last 5):")
        try:
            dates = stock.earnings_dates
            if dates is not None and not dates.empty:
                print(dates.head(5).to_string(indent=2))
            else:
                print("  None or empty")
        except Exception as e:
            print(f"  Error: {e}")

        # Earnings history
        print("\nEarnings History (last 3):")
        try:
            hist = stock.earnings_history
            if hist is not None and not hist.empty:
                print(hist.head(3).to_string(indent=2))
            else:
                print("  None or empty")
        except Exception as e:
            print(f"  Error: {e}")

    except Exception as e:
        print(f"  YFinance error: {e}")

    # Step 2: Check our earnings_info
    print("\n2. EARNINGS INFO (from windows.py)")
    print("-" * 40)

    try:
        from src.analytics.earnings_intelligence.windows import get_earnings_info

        info = get_earnings_info(ticker)
        print(f"  earnings_date: {info.earnings_date if info else 'N/A'}")
        print(f"  days_to_earnings: {info.days_to_earnings if info else 'N/A'}")
        print(f"  in_compute_window: {info.in_compute_window if info else 'N/A'}")
        print(f"  in_action_window: {info.in_action_window if info else 'N/A'}")
        print(f"  source: {info.source if info else 'N/A'}")
    except Exception as e:
        print(f"  Error: {e}")

    # Step 3: Check IES calculation
    print("\n3. IES CALCULATION")
    print("-" * 40)

    try:
        from src.analytics.earnings_intelligence.ies_calculator import calculate_ies

        ies_result = calculate_ies(ticker)
        print(f"  IES: {ies_result.ies:.1f}" if ies_result.ies else "  IES: None")
        print(f"  Regime: {ies_result.regime.value if ies_result.regime else 'None'}")
        print(f"  Data Quality: {ies_result.data_quality.value if ies_result.data_quality else 'None'}")
        print(f"  Missing Inputs: {ies_result.missing_inputs}")
    except Exception as e:
        print(f"  Error: {e}")

    # Step 4: Check EQS/ECS calculation
    print("\n4. EQS/ECS CALCULATION")
    print("-" * 40)

    try:
        from src.analytics.earnings_intelligence.eqs_calculator import calculate_eqs
        from src.analytics.earnings_intelligence.ecs_calculator import calculate_ecs

        eqs_result = calculate_eqs(ticker)
        print(f"  EQS: {eqs_result.eqs:.1f}" if eqs_result.eqs else "  EQS: None")
        print(f"  Event Z: {eqs_result.event_z:.2f}" if eqs_result.event_z else "  Event Z: None")
        print(
            f"  EPS Surprise: {eqs_result.eps_surprise_pct:+.1f}%" if eqs_result.eps_surprise_pct else "  EPS Surprise: None")

        ecs_result = calculate_ecs(ticker)
        print(f"  ECS Category: {ecs_result.ecs_category.value if ecs_result.ecs_category else 'None'}")
        print(f"  Cleared Bar: {ecs_result.cleared_bar}")
        print(f"  Score Adjustment: {ecs_result.score_adjustment:+d}")
    except Exception as e:
        print(f"  Error: {e}")

    # Step 5: Check AI context
    print("\n5. AI CONTEXT (what chat.py receives)")
    print("-" * 40)

    try:
        from src.analytics.earnings_intelligence.chat_integration import get_earnings_for_ai

        context = get_earnings_for_ai(ticker)
        if context:
            print(context)
        else:
            print("  [Empty context returned]")
    except Exception as e:
        print(f"  Error: {e}")

    # Step 6: Check database (if available)
    print("\n6. DATABASE CHECK")
    print("-" * 40)

    try:
        from src.db.connection import get_engine
        import pandas as pd

        engine = get_engine()

        # Check earnings_analysis table
        query = f"""
            SELECT ticker, earnings_date, overall_sentiment, eps_surprise_pct, 
                   guidance_direction, score_adjustment
            FROM earnings_analysis
            WHERE ticker = '{ticker}'
            ORDER BY earnings_date DESC
            LIMIT 3
        """
        df = pd.read_sql(query, engine)
        if not df.empty:
            print("  earnings_analysis table:")
            print(df.to_string(indent=4))
        else:
            print("  No data in earnings_analysis table")

        # Check earnings_intelligence table
        query = f"""
            SELECT ticker, earnings_date, ies, eqs, ecs, regime
            FROM earnings_intelligence
            WHERE ticker = '{ticker}'
            ORDER BY earnings_date DESC
            LIMIT 3
        """
        try:
            df = pd.read_sql(query, engine)
            if not df.empty:
                print("\n  earnings_intelligence table:")
                print(df.to_string(indent=4))
            else:
                print("\n  No data in earnings_intelligence table")
        except:
            print("\n  earnings_intelligence table not found (expected if not created)")

    except Exception as e:
        print(f"  Database not available: {e}")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "MU"

    diagnose_ticker(ticker)