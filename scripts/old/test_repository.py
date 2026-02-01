"""
Test Repository

Run this script to verify database operations work.

Usage:
    python scripts/test_repository.py
"""

import sys
import os
from datetime import date

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.db.repository import Repository


def main():
    print("=" * 50)
    print("Alpha Platform - Repository Test")
    print("=" * 50)
    print()

    repo = Repository()

    # Test 1: Get universe
    print("1. Testing get_universe()...")
    tickers = repo.get_universe()
    print(f"   Found {len(tickers)} tickers in universe")
    print(f"   First 5: {tickers[:5]}")
    print()

    # Test 2: Save and get screener score
    print("2. Testing save_screener_score()...")
    test_scores = {
        'sentiment_score': 65,
        'fundamental_score': 72,
        'growth_score': 68,
        'dividend_score': 45,
        'technical_score': 70,
        'gap_score': 55,
        'likelihood_score': 67,
        'total_score': 68,
    }
    repo.save_screener_score('TEST', date.today(), test_scores)
    print("   Saved test score for 'TEST' ticker")
    print()

    # Test 3: Get latest scores
    print("3. Testing get_latest_scores()...")
    scores_df = repo.get_latest_scores(limit=5)
    print(f"   Found {len(scores_df)} scores")
    if len(scores_df) > 0:
        print(f"   Columns: {list(scores_df.columns)[:5]}...")
    print()

    # Test 4: Save and get signal
    print("4. Testing save_signal()...")
    test_signal = {
        'type': 'BUY',
        'strength': 4,
        'color': '#4ade80',
        'reasons': ['Test signal'],
        'sentiment_score': 65,
        'fundamental_score': 72,
    }
    repo.save_signal('TEST', date.today(), test_signal)
    print("   Saved test signal for 'TEST' ticker")
    print()

    # Test 5: Get system status
    print("5. Testing get_system_status()...")
    status_df = repo.get_system_status()
    print(f"   Components: {list(status_df['component'])}")
    print()

    # Test 6: Update system status
    print("6. Testing update_system_status()...")
    repo.update_system_status('screener', 'running', progress_pct=50, progress_message='Testing...')
    status_df = repo.get_system_status()
    screener_status = status_df[status_df['component'] == 'screener'].iloc[0]
    print(f"   Screener status: {screener_status['status']}, progress: {screener_status['progress_pct']}%")

    # Reset status
    repo.update_system_status('screener', 'stopped')
    print()

    print("=" * 50)
    print("SUCCESS - All repository tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()