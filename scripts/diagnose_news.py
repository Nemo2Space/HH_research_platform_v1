"""
Diagnostic script to check news articles in the database
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load .env file
from dotenv import load_dotenv
env_path = os.path.join(project_root, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)

import pandas as pd
from src.db.connection import get_engine

engine = get_engine()

print("=" * 70)
print("NEWS ARTICLES DIAGNOSTIC")
print("=" * 70)

# Check recent articles
query = """
    SELECT 
        id, 
        ticker, 
        LEFT(headline, 50) as headline,
        source,
        published_at,
        fetched_at,
        created_at
    FROM news_articles 
    ORDER BY created_at DESC 
    LIMIT 20
"""

df = pd.read_sql(query, engine)

print(f"\nFound {len(df)} recent articles:\n")
print(df.to_string())

# Check stats
print("\n" + "=" * 70)
print("STATISTICS")
print("=" * 70)

stats_query = """
    SELECT 
        COUNT(*) as total_articles,
        COUNT(published_at) as with_published_at,
        COUNT(*) - COUNT(published_at) as missing_published_at,
        COUNT(DISTINCT ticker) as unique_tickers
    FROM news_articles
"""

stats = pd.read_sql(stats_query, engine)
print(f"\nTotal articles: {stats['total_articles'].iloc[0]}")
print(f"With published_at: {stats['with_published_at'].iloc[0]}")
print(f"Missing published_at: {stats['missing_published_at'].iloc[0]}")
print(f"Unique tickers: {stats['unique_tickers'].iloc[0]}")

# Check sample of what published_at looks like when present
print("\n" + "=" * 70)
print("SAMPLE OF ARTICLES WITH DATES")
print("=" * 70)

sample_query = """
    SELECT 
        ticker,
        LEFT(headline, 40) as headline,
        published_at,
        source
    FROM news_articles 
    WHERE published_at IS NOT NULL
    ORDER BY published_at DESC
    LIMIT 10
"""

sample = pd.read_sql(sample_query, engine)
if sample.empty:
    print("\nNO ARTICLES HAVE published_at SET!")
else:
    print(f"\n{sample.to_string()}")

# Check what sources we have
print("\n" + "=" * 70)
print("ARTICLES BY SOURCE")
print("=" * 70)

source_query = """
    SELECT 
        COALESCE(source, 'Unknown') as source,
        COUNT(*) as count,
        COUNT(published_at) as with_date
    FROM news_articles
    GROUP BY source
    ORDER BY count DESC
    LIMIT 15
"""

sources = pd.read_sql(source_query, engine)
print(f"\n{sources.to_string()}")