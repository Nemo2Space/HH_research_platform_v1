"""Test Gap Analysis"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.screener.gap_analysis import test_gap_analysis

if __name__ == "__main__":
    test_gap_analysis()