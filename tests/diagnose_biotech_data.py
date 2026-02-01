"""
Diagnose Biotech Data Coverage
==============================
Identifies exactly which biotech tickers have AI data vs missing data.

Run: python diagnose_biotech_data.py
"""

import sys

sys.path.insert(0, '..')
import pandas as pd
from datetime import datetime, date

BIOTECH_TICKERS = [
    "ADPT", "DYN", "CERT", "COGT", "SRPT", "MNKD", "CDTX", "BCRX", "ARDX", "TXG",
    "VRDN", "TWST", "EWTX", "AVDL", "JANX", "SDGR", "ABCL", "NVAX", "STOK", "AMLX",
    "NTLA", "PCRX", "XERS", "GPCR", "NEOG", "ELVN", "BBNX", "PGEN", "QURE", "WVE",
    "OPK", "DVAX", "ORIC", "MDXG", "TRVI", "ATAI", "SPRY", "CRMD", "FTRE", "ABUS",
    "SANA", "TSHA", "PHAT", "IOVA", "GERN", "AVXL", "IMNM", "GOSS", "AKBA", "SVRA",
    "PROK", "DAWN", "TNGX", "KURA", "KALV", "VIR", "NRIX", "RLAY", "MRVI", "MYGN",
    "RZLT", "TERN", "CMPX", "AQST", "VSTM", "ATYR", "ESPR", "PRME", "PSNL", "SRDX",
    "XOMA", "REPL", "ERAS", "CRVS", "ATXS", "LXRX", "ALT", "ALDX", "ABEO", "CTMX",
    "OCGN", "LRMR", "RCKT", "PACB", "IMRX", "AUTL", "FULC", "ABSI"
]


def get_db_connection():
    try:
        from src.db.connection import get_connection
        cm = get_connection()
        return cm.__enter__()
    except:
        import psycopg2
        return psycopg2.connect(
            host="localhost", port=5432, dbname="alpha_platform",
            user="alpha", password="alpha_secure_2024"
        )


print("=" * 100)
print("BIOTECH DATA COVERAGE DIAGNOSTIC")
print("=" * 100)
print(f"\nTotal biotech tickers in list: {len(BIOTECH_TICKERS)}")

conn = get_db_connection()

# =============================================================================
# 1. CHECK WHICH BIOTECHS ARE IN screener_scores (BASE TABLE)
# =============================================================================
print("\n" + "=" * 100)
print("1. SCREENER_SCORES (Base Table)")
print("=" * 100)

with conn.cursor() as cur:
    cur.execute("""
        SELECT DISTINCT ticker 
        FROM screener_scores 
        WHERE ticker = ANY(%s)
    """, (BIOTECH_TICKERS,))
    in_screener = [r[0] for r in cur.fetchall()]

missing_from_screener = set(BIOTECH_TICKERS) - set(in_screener)
print(f"\n✓ Biotechs in screener_scores: {len(in_screener)}/{len(BIOTECH_TICKERS)}")
if missing_from_screener:
    print(f"✗ Missing from screener_scores ({len(missing_from_screener)}): {sorted(missing_from_screener)}")

# =============================================================================
# 2. CHECK AI DATA COVERAGE FOR EACH TABLE
# =============================================================================
print("\n" + "=" * 100)
print("2. AI DATA COVERAGE BY TABLE")
print("=" * 100)

ai_tables = {
    'ai_analysis': ('ticker', 'ai_action'),
    'ai_recommendations': ('ticker', 'ai_probability'),
    'committee_decisions': ('ticker', 'verdict'),
    'agent_votes': ('ticker', 'agent_role'),
    'alpha_predictions': ('ticker', 'predicted_probability'),
    'enhanced_scores': ('ticker', 'insider_score'),
    'trading_signals': ('ticker', 'signal_type'),
    'fda_calendar': ('ticker', 'drug_name'),
    'earnings_calendar': ('ticker', 'earnings_date'),
}

coverage_data = {}

for table, (ticker_col, check_col) in ai_tables.items():
    with conn.cursor() as cur:
        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            )
        """, (table,))
        exists = cur.fetchone()[0]

        if not exists:
            print(f"\n❌ {table}: Table does not exist")
            coverage_data[table] = {'exists': False, 'tickers': []}
            continue

        # Get biotechs in this table
        cur.execute(f"""
            SELECT DISTINCT {ticker_col}
            FROM {table}
            WHERE {ticker_col} = ANY(%s)
        """, (BIOTECH_TICKERS,))
        tickers_in_table = [r[0] for r in cur.fetchall()]

        coverage_data[table] = {'exists': True, 'tickers': tickers_in_table}

        pct = len(tickers_in_table) / len(BIOTECH_TICKERS) * 100
        print(
            f"\n{'✓' if pct > 50 else '⚠' if pct > 10 else '✗'} {table}: {len(tickers_in_table)}/{len(BIOTECH_TICKERS)} ({pct:.0f}%)")

        if tickers_in_table:
            print(f"   Has data: {sorted(tickers_in_table)[:15]}{'...' if len(tickers_in_table) > 15 else ''}")

        missing = set(BIOTECH_TICKERS) - set(tickers_in_table)
        if missing:
            print(f"   Missing:  {sorted(missing)[:15]}{'...' if len(missing) > 15 else ''}")

# =============================================================================
# 3. FIND BIOTECHS WITH COMPLETE AI DATA
# =============================================================================
print("\n" + "=" * 100)
print("3. BIOTECHS WITH COMPLETE vs PARTIAL AI DATA")
print("=" * 100)

# Key AI tables we need
key_tables = ['ai_analysis', 'committee_decisions', 'trading_signals', 'enhanced_scores']

# Find biotechs with ALL key data
biotechs_with_all = set(BIOTECH_TICKERS)
for table in key_tables:
    if coverage_data.get(table, {}).get('exists'):
        biotechs_with_all &= set(coverage_data[table]['tickers'])

# Find biotechs with ANY data
biotechs_with_any = set()
for table in key_tables:
    if coverage_data.get(table, {}).get('exists'):
        biotechs_with_any |= set(coverage_data[table]['tickers'])

# Biotechs with no AI data at all
biotechs_no_data = set(BIOTECH_TICKERS) - biotechs_with_any

print(f"\n✓ Complete AI data (all 4 key tables): {len(biotechs_with_all)}")
if biotechs_with_all:
    print(f"   {sorted(biotechs_with_all)}")

print(f"\n⚠ Partial AI data (at least 1 table): {len(biotechs_with_any - biotechs_with_all)}")
partial = biotechs_with_any - biotechs_with_all
if partial:
    print(f"   {sorted(partial)}")

print(f"\n✗ No AI data at all: {len(biotechs_no_data)}")
if biotechs_no_data:
    print(f"   {sorted(biotechs_no_data)}")

# =============================================================================
# 4. DETAILED BREAKDOWN PER TICKER
# =============================================================================
print("\n" + "=" * 100)
print("4. DETAILED DATA MATRIX")
print("=" * 100)

# Build matrix
matrix_data = []
for ticker in sorted(BIOTECH_TICKERS):
    row = {'ticker': ticker}
    row['in_screener'] = ticker in in_screener
    for table in key_tables + ['fda_calendar', 'ai_recommendations', 'alpha_predictions']:
        if coverage_data.get(table, {}).get('exists'):
            row[table] = ticker in coverage_data[table]['tickers']
        else:
            row[table] = False
    matrix_data.append(row)

df_matrix = pd.DataFrame(matrix_data)

# Print summary
print(f"\n{'Ticker':<8} {'Screener':>8} {'AI_Anal':>8} {'Committ':>8} {'Signals':>8} {'Enhanced':>8} {'FDA':>8}")
print("-" * 70)
for _, row in df_matrix.iterrows():
    vals = [
        row['ticker'],
        '✓' if row['in_screener'] else '✗',
        '✓' if row.get('ai_analysis', False) else '✗',
        '✓' if row.get('committee_decisions', False) else '✗',
        '✓' if row.get('trading_signals', False) else '✗',
        '✓' if row.get('enhanced_scores', False) else '✗',
        '✓' if row.get('fda_calendar', False) else '✗',
    ]
    print(f"{vals[0]:<8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {vals[4]:>8} {vals[5]:>8} {vals[6]:>8}")

# =============================================================================
# 5. RECOMMENDATIONS
# =============================================================================
print("\n" + "=" * 100)
print("5. RECOMMENDATIONS")
print("=" * 100)

# Count needs
needs_ai_analysis = len(set(BIOTECH_TICKERS) - set(coverage_data.get('ai_analysis', {}).get('tickers', [])))
needs_committee = len(set(BIOTECH_TICKERS) - set(coverage_data.get('committee_decisions', {}).get('tickers', [])))
needs_signals = len(set(BIOTECH_TICKERS) - set(coverage_data.get('trading_signals', {}).get('tickers', [])))
needs_enhanced = len(set(BIOTECH_TICKERS) - set(coverage_data.get('enhanced_scores', {}).get('tickers', [])))
needs_fda = len(set(BIOTECH_TICKERS) - set(coverage_data.get('fda_calendar', {}).get('tickers', [])))

print(f"""
Data Population Needed:
───────────────────────────────────────────────────
│ Table                  │ Biotechs Missing │ Priority │
│────────────────────────│──────────────────│──────────│
│ ai_analysis            │ {needs_ai_analysis:>16} │ HIGH     │
│ committee_decisions    │ {needs_committee:>16} │ HIGH     │
│ trading_signals        │ {needs_signals:>16} │ HIGH     │
│ enhanced_scores        │ {needs_enhanced:>16} │ MEDIUM   │
│ fda_calendar           │ {needs_fda:>16} │ HIGH     │
───────────────────────────────────────────────────

Run `python populate_biotech_ai_data.py` to fill these gaps.
""")

# =============================================================================
# 6. EXPORT MISSING TICKERS FOR POPULATION SCRIPT
# =============================================================================
print("\n" + "=" * 100)
print("6. EXPORTING LISTS FOR POPULATION")
print("=" * 100)

# Create export dict
export_data = {
    'all_biotechs': BIOTECH_TICKERS,
    'in_screener': in_screener,
    'missing_screener': list(missing_from_screener),
    'needs_ai_analysis': list(set(BIOTECH_TICKERS) - set(coverage_data.get('ai_analysis', {}).get('tickers', []))),
    'needs_committee': list(
        set(BIOTECH_TICKERS) - set(coverage_data.get('committee_decisions', {}).get('tickers', []))),
    'needs_signals': list(set(BIOTECH_TICKERS) - set(coverage_data.get('trading_signals', {}).get('tickers', []))),
    'needs_enhanced': list(set(BIOTECH_TICKERS) - set(coverage_data.get('enhanced_scores', {}).get('tickers', []))),
    'needs_fda': list(set(BIOTECH_TICKERS) - set(coverage_data.get('fda_calendar', {}).get('tickers', []))),
}

import json

with open('biotech_data_gaps.json', 'w') as f:
    json.dump(export_data, f, indent=2)

print(f"✓ Exported data gaps to biotech_data_gaps.json")

conn.close()

print("\n" + "=" * 100)
print("DIAGNOSTIC COMPLETE")
print("=" * 100)