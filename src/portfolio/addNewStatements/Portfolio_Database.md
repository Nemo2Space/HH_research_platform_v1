# Portfolio Database Management Guide

## Overview

This guide covers how to manage your portfolio data in the PostgreSQL database, including initial setup, importing IBKR statements, incremental updates, and troubleshooting.

---

## Table of Contents

1. [File Structure](#file-structure)
2. [Initial Setup](#initial-setup)
3. [Importing IBKR Statements](#importing-ibkr-statements)
4. [Incremental Updates](#incremental-updates)
5. [Verification & Troubleshooting](#verification--troubleshooting)
6. [Database Schema](#database-schema)
7. [Common Workflows](#common-workflows)
8. [FAQ](#faq)

---

## File Structure

```
HH_research_platform_v1/
├── src/
│   └── db/
│       ├── connection.py          # Database connection management
│       └── portfolio_db.py        # Portfolio repository (CRUD operations)
├── scripts/
│   ├── import_ibkr_csv.py         # Import IBKR CSV statements
│   ├── verify_portfolio_db.py     # Verify and cleanup database
│   └── check_db.py                # Quick database summary
├── dashboard/
│   └── portfolio_tab.py           # Streamlit portfolio UI
└── data/
    └── statements/                # Store your IBKR CSV files here
```

---

## Initial Setup

### Prerequisites

1. **PostgreSQL database** running (via TimescaleDB or standalone)
2. **Python environment** with required packages:
   ```bash
   pip install psycopg2-binary pandas sqlalchemy
   ```

### Step 1: Initialize Database Schema

The schema is automatically created when you first import data. However, if you need to manually initialize:

```python
from src.db.portfolio_db import PortfolioRepository

repo = PortfolioRepository()
repo.init_schema()
print("Schema initialized!")
```

### Step 2: Verify Database Connection

```bash
python scripts/check_db.py
```

Expected output (empty database):
```
============================================================
  Accounts in Database
============================================================
  (no accounts found)
============================================================
```

---

## Importing IBKR Statements

### How to Export from IBKR

1. Log into **IBKR Account Management** (Client Portal)
2. Navigate to: **Reports → Statements**
3. Select:
   - **Statement Type:** Activity
   - **Period:** Custom (select your date range)
   - **Format:** CSV
4. For multi-account:
   - Select **"All Accounts"** or create a consolidated report
   - The file will be named something like `U17994267_MULTI_*.csv`
5. Download and save to `data/statements/`

### Import Commands

#### Import a Single Statement
```bash
python scripts/import_ibkr_csv.py data/statements/YOUR_FILE.csv
```

#### Import Multiple Files
```bash
python scripts/import_ibkr_csv.py data/statements/U20993660_*.csv
python scripts/import_ibkr_csv.py data/statements/MULTI_*.csv
```

### Import Output Example

```
============================================================
  IBKR Statement Import
============================================================
  File: U20993660_20260124.csv
  Account: U20993660
  Period: 2025-03-21 to 2026-01-23

  Importing data...
    ✓ Account created/updated
    ✓ NAV: $753,014.87
    ✓ Deposits/Withdrawals: 2 records
    ✓ Dividends: 86 records
    ✓ Interest: 6 records
    ✓ Withholding Tax: 81 records
    ✓ Trades: 872 records
    ✓ Positions: 100 records

  ✅ Import completed successfully!
============================================================
```

### Important Notes on Multi-Account Imports

When importing a **MULTI/Consolidated CSV**:
- All deposits/withdrawals are assigned to the primary account (U17994267)
- Internal transfers between accounts are recorded
- The system automatically handles self-transfer artifacts

**Recommended approach:**
1. Import the MULTI CSV first (contains all consolidated data)
2. Import individual account CSVs if you have them (for separate position snapshots)

---

## Incremental Updates

### Weekly/Monthly Update Process

#### Step 1: Export New Activity from IBKR

1. Go to IBKR → Reports → Statements
2. Select **Activity Statement**
3. Set date range: **[Last import date] → [Today]**
4. Export as CSV

#### Step 2: Import New Data

```bash
python scripts/import_ibkr_csv.py data/statements/NEW_STATEMENT.csv
```

#### Step 3: Verify Update

```bash
python scripts/verify_portfolio_db.py check
```

### What Gets Updated

| Data Type | Behavior |
|-----------|----------|
| **NAV History** | New date added; historical dates preserved |
| **Positions** | Latest snapshot replaces previous (per date) |
| **Trades** | Appended; duplicates ignored |
| **Dividends** | Appended; duplicates ignored |
| **Deposits/Withdrawals** | Appended; duplicates ignored |
| **Interest** | Appended; duplicates ignored |
| **Fees** | Appended; duplicates ignored |

### Deduplication Logic

The import script prevents duplicate entries using these keys:

- **Trades:** `account_id + date + symbol + quantity + price`
- **Dividends:** `account_id + date + symbol + amount`
- **Deposits:** `account_id + date + description + amount`
- **Interest:** `account_id + date + amount`

---

## Verification & Troubleshooting

### Quick Database Check

```bash
python scripts/check_db.py
```

Output:
```
============================================================
  Accounts in Database
============================================================
  - U17994267: Stichting Legal Owner Keheilan Fund
  - U20993660: Stichting Legal Owner Keheilan Fund

U20993660:
  NAV: $753,014.87
  Deposits: $717,000.00
  True Profit: $36,014.87 (5.02%)
  Dividends: $475.79
  Interest: $2,696.81
  Withholding Tax: $-69.29

U17994267:
  NAV: $1,455,292.52
  Deposits: $1,414,226.96
  True Profit: $41,065.56 (2.90%)
  ...
============================================================
```

### Detailed Verification

```bash
python scripts/verify_portfolio_db.py check
```

This shows:
- All deposits/withdrawals with transfer classification
- Internal vs external deposit breakdown
- Per-account True Profit calculation
- Consolidated totals

### Cleanup Database (Start Fresh)

⚠️ **WARNING:** This deletes ALL portfolio data!

```bash
python scripts/verify_portfolio_db.py cleanup
```

You'll be prompted to type `YES` to confirm.

### Verify Expected Values

```bash
python scripts/verify_portfolio_db.py verify
```

Compares database values against expected calculations.

---

## Database Schema

### Tables

| Table | Description |
|-------|-------------|
| `portfolio_accounts` | Account information (ID, name, type, currency) |
| `portfolio_nav_history` | Daily NAV snapshots |
| `portfolio_positions` | Current holdings per date |
| `portfolio_trades` | All trade history |
| `portfolio_dividends` | Dividend payments |
| `portfolio_interest` | Interest income |
| `portfolio_withholding_tax` | Tax withholdings |
| `portfolio_deposits_withdrawals` | Cash movements |
| `portfolio_fees` | Fees and commissions |
| `portfolio_import_history` | Import audit trail |

### Key Relationships

```
portfolio_accounts (1) ──── (N) portfolio_nav_history
                    (1) ──── (N) portfolio_positions
                    (1) ──── (N) portfolio_trades
                    (1) ──── (N) portfolio_dividends
                    (1) ──── (N) portfolio_deposits_withdrawals
```

---

## Common Workflows

### Workflow 1: First-Time Setup

```bash
# 1. Verify database connection
python scripts/check_db.py

# 2. Import your historical IBKR statements
python scripts/import_ibkr_csv.py data/statements/MULTI_inception_to_today.csv

# 3. Verify import
python scripts/verify_portfolio_db.py check

# 4. Launch dashboard
streamlit run dashboard/app.py
```

### Workflow 2: Weekly Update

```bash
# 1. Download new IBKR statement (last week's activity)
# Save to: data/statements/U17994267_20260131.csv

# 2. Import
python scripts/import_ibkr_csv.py data/statements/U17994267_20260131.csv

# 3. Quick verify
python scripts/check_db.py
```

### Workflow 3: Fix Data Issues

```bash
# 1. Check current state
python scripts/verify_portfolio_db.py check

# 2. If data is corrupted, cleanup
python scripts/verify_portfolio_db.py cleanup

# 3. Re-import from scratch
python scripts/import_ibkr_csv.py data/statements/FULL_HISTORY.csv

# 4. Verify
python scripts/verify_portfolio_db.py check
```

### Workflow 4: Add New Account

```bash
# 1. Export statement for new account from IBKR
# 2. Import it
python scripts/import_ibkr_csv.py data/statements/NEW_ACCOUNT.csv

# 3. Verify it appears
python scripts/check_db.py
```

---

## FAQ

### Q: Why does U17994267 show "Transfer In From Account U17994267"?

**A:** This is an artifact from importing MULTI/consolidated CSV files. The MULTI CSV contains transactions from all accounts, and some transfer records appear duplicated. The system automatically ignores these "self-transfers" in calculations.

### Q: Why is my True Profit negative for one account but positive for consolidated?

**A:** If you imported a MULTI CSV, all external deposits are assigned to one account (typically U17994267), while the NAV is split between accounts. The **consolidated view ("All")** shows the correct total picture.

### Q: How do I handle internal transfers between accounts?

**A:** The system automatically:
- Detects internal transfers (e.g., "Transfer Out To Account U20993660")
- Excludes them from external deposit calculations
- Includes them in per-account capital allocation

### Q: Can I import the same file twice?

**A:** Yes! The deduplication logic prevents duplicate entries. Only new records will be added.

### Q: How do I see historical NAV?

```python
from src.db.portfolio_db import PortfolioRepository
repo = PortfolioRepository()

# Get NAV history for an account
nav_history = repo.get_nav_history("U20993660")
print(nav_history)
```

### Q: How do I export data from the database?

```python
from src.db.portfolio_db import PortfolioRepository
import pandas as pd

repo = PortfolioRepository()

# Export positions to CSV
positions = repo.get_positions("U20993660")
positions.to_csv("my_positions.csv", index=False)

# Export trades
trades = repo.get_trades("U20993660")
trades.to_csv("my_trades.csv", index=False)
```

### Q: What if IBKR changes their CSV format?

The parser handles standard IBKR Activity Statement format. If IBKR changes their format significantly, update the `IBKRStatementParser` class in `scripts/import_ibkr_csv.py`.

---

## Support Files Reference

| File | Purpose | Command |
|------|---------|---------|
| `scripts/import_ibkr_csv.py` | Import IBKR CSV files | `python scripts/import_ibkr_csv.py <file>` |
| `scripts/verify_portfolio_db.py` | Verify/cleanup database | `python scripts/verify_portfolio_db.py check` |
| `scripts/check_db.py` | Quick database summary | `python scripts/check_db.py` |
| `src/db/portfolio_db.py` | Database repository class | (imported in code) |
| `src/db/connection.py` | Database connection | (imported in code) |
| `dashboard/portfolio_tab.py` | Streamlit portfolio UI | `streamlit run dashboard/app.py` |

---

## Quick Reference Card

```bash
# === DAILY/WEEKLY ===
python scripts/check_db.py                              # Quick status

# === IMPORTING ===
python scripts/import_ibkr_csv.py <file.csv>           # Import statement

# === VERIFICATION ===
python scripts/verify_portfolio_db.py check            # Detailed check
python scripts/verify_portfolio_db.py verify           # Validate values

# === MAINTENANCE ===
python scripts/verify_portfolio_db.py cleanup          # ⚠️ Delete all data

# === DASHBOARD ===
streamlit run dashboard/app.py                          # Launch UI
```

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-24 | 1.0 | Initial database integration |
| 2026-01-24 | 1.1 | Added self-transfer handling for MULTI CSVs |
| 2026-01-24 | 1.2 | Fixed per-account capital calculation |

---

*Last updated: January 24, 2026*