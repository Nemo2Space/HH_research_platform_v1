"""
Alpha Platform - Quick Stock Manager

Edit the lists below and run this file from PyCharm.
Everything happens automatically!

Instructions:
1. Edit ADD_STOCKS or REMOVE_STOCKS lists below
2. Run this file (Shift+F10 in PyCharm)
3. Wait for completion
4. Refresh your dashboard
"""

# ============================================================================
# EDIT THESE LISTS
# ============================================================================

# Stocks to ADD (leave empty [] if none)
ADD_STOCKS = [
 "DELL"
    # "IBM",
    # "QCOM",
]

# Stocks to REMOVE (leave empty [] if none)
REMOVE_STOCKS = [
    # "COIN",
    # "SQ",
]

# Run full screener after changes? (generates signals for all stocks)
RUN_SCREENER = True

# ============================================================================
# DON'T EDIT BELOW THIS LINE
# ============================================================================

import os
import sys
import subprocess

# Setup path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    print("\n" + "=" * 60)
    print("🚀 ALPHA PLATFORM - STOCK MANAGER")
    print("=" * 60)

    changes_made = False
    added = []

    # Process removals first
    if REMOVE_STOCKS:
        print(f"\n🗑️  Removing {len(REMOVE_STOCKS)} stock(s)...")
        for ticker in REMOVE_STOCKS:
            print(f"   Removing {ticker}...")
            subprocess.run([
                sys.executable,
                os.path.join(project_root, "scripts", "manage_universe.py"),
                "--remove", ticker
            ], cwd=project_root)
        changes_made = True

    # Process additions
    if ADD_STOCKS:
        print(f"\n➕ Adding {len(ADD_STOCKS)} stock(s)...")
        added = []
        for ticker in ADD_STOCKS:
            ticker = ticker.upper().strip()

            # Check if already exists BEFORE calling subprocess
            import pandas as pd
            universe_file = os.path.join(project_root, 'config', 'universe.csv')
            df = pd.read_csv(universe_file)

            if ticker in df['ticker'].values:
                print(f"   ⚠️  {ticker} already in universe - skipping")
                continue

            print(f"\n   Adding {ticker}...")
            subprocess.run([
                sys.executable,
                os.path.join(project_root, "scripts", "manage_universe.py"),
                "--add", ticker
            ], cwd=project_root)
            added.append(ticker)

        if added:
            changes_made = True
        else:
            print("\n⚠️  All stocks already exist in universe.")



    # Run screener only for added stocks (not all 100!)
    # Run screener only for added stocks (not all 100!)
    if RUN_SCREENER and added:
        print("\n" + "=" * 60)
        print(f"📊 Running screener for {len(added)} new stock(s)...")
        print("=" * 60)

        subprocess.run([
                           sys.executable,
                           os.path.join(project_root, "scripts", "run_full_screener.py"),
                           "--ticker"] + added,
                       cwd=project_root
                       )
    elif RUN_SCREENER and ADD_STOCKS and not added:
        print("\n⚠️  No new stocks to process. Exiting.")

    if not ADD_STOCKS and not REMOVE_STOCKS:
        print("\n⚠️  No changes made. Add tickers to ADD_STOCKS or REMOVE_STOCKS.")
        print("   Edit the lists at the top of this file.")

    print("\n" + "=" * 60)
    print("✅ COMPLETE! Refresh your dashboard to see changes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
