"""Test AI Chat module."""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from src.ai.chat import AlphaChat

def main():
    print("=" * 60)
    print("AI CHAT TEST")
    print("=" * 60)

    chat = AlphaChat()

    if not chat.available:
        print("ERROR: AI Chat not available. Check Qwen server.")
        return

    print("✅ Connected to Qwen\n")

    # Test 1: General question
    print("1. General question about backtest insights...")
    response = chat.chat("What are the best performing strategies based on the backtest data?")
    print(f"   Response: {response[:300]}...\n")

    # Test 2: Ticker-specific question
    print("2. Asking about AAPL...")
    response = chat.ask_about_ticker("AAPL")
    print(f"   Response: {response[:300]}...\n")

    # Test 3: Get recommendation
    print("3. Getting recommendation for NVDA...")
    response = chat.get_recommendation("NVDA")
    print(f"   Response: {response[:400]}...\n")

    print("=" * 60)
    print("SUCCESS - AI Chat working!")
    print("=" * 60)

if __name__ == "__main__":
    main()