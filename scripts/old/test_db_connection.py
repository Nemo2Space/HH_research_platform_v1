"""
Test Database Connection

Run this script to verify the database connection works.

Usage:
    python scripts/test_db_connection.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

# Load .env file
load_dotenv()

from src.db.connection import test_connection, DatabaseConfig


def main():
    print("=" * 50)
    print("Alpha Platform - Database Connection Test")
    print("=" * 50)
    print()
    
    # Show config (hide password)
    config = DatabaseConfig.from_env()
    print(f"Host:     {config.host}")
    print(f"Port:     {config.port}")
    print(f"Database: {config.database}")
    print(f"User:     {config.user}")
    print(f"Password: {'*' * len(config.password) if config.password else '(not set)'}")
    print()
    
    if not config.password:
        print("ERROR: POSTGRES_PASSWORD not set!")
        print("Please set it in your .env file")
        sys.exit(1)
    
    try:
        test_connection(config)
        print()
        print("=" * 50)
        print("SUCCESS - Database connection working!")
        print("=" * 50)
    except Exception as e:
        print()
        print("=" * 50)
        print(f"FAILED - {e}")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()
