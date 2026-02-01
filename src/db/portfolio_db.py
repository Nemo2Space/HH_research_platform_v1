"""
Portfolio Database Module
=========================

Stores portfolio data in PostgreSQL/TimescaleDB for fast access and historical tracking.
Integrates with existing src.db.connection module.

Tables:
- portfolio_accounts: Account information
- portfolio_deposits_withdrawals: Cash flows (deposits, withdrawals, transfers)
- portfolio_dividends: Dividend payments
- portfolio_interest: Interest income
- portfolio_withholding_tax: Tax withholdings
- portfolio_trades: Trade history
- portfolio_fees: Fees and commissions
- portfolio_nav_history: Daily NAV snapshots
- portfolio_positions: Current positions (updated on each import)

Usage:
    from src.db.portfolio_db import PortfolioRepository

    repo = PortfolioRepository()
    repo.import_statement(parsed_statement)  # Import from CSV

    # Get account data
    summary = repo.get_account_summary(account_id)
"""

import os
import logging
from datetime import datetime, date
from typing import List, Dict, Optional, Any
import hashlib
import pandas as pd

# Use existing connection module
try:
    from src.db.connection import get_connection, get_engine
except ImportError:
    # Fallback if running standalone
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv()

    from contextlib import contextmanager

    @contextmanager
    def get_connection():
        """Fallback connection for standalone use."""
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "alpha_platform"),
            user=os.getenv("POSTGRES_USER", "alpha"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_engine():
        from sqlalchemy import create_engine
        conn_str = f"postgresql://{os.getenv('POSTGRES_USER', 'alpha')}:{os.getenv('POSTGRES_PASSWORD', '')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'alpha_platform')}"
        return create_engine(conn_str)

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA
# =============================================================================

SCHEMA_SQL = """
-- Portfolio Accounts
CREATE TABLE IF NOT EXISTS portfolio_accounts (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255),
    account_type VARCHAR(100),
    base_currency VARCHAR(10) DEFAULT 'USD',
    is_consolidated BOOLEAN DEFAULT FALSE,
    accounts_included TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Deposits & Withdrawals
CREATE TABLE IF NOT EXISTS portfolio_deposits_withdrawals (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    description TEXT,
    amount DECIMAL(18, 2) NOT NULL,
    currency VARCHAR(10) DEFAULT 'USD',
    record_hash VARCHAR(64) UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_dw_account_date ON portfolio_deposits_withdrawals(account_id, date);

-- Dividends
CREATE TABLE IF NOT EXISTS portfolio_dividends (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    symbol VARCHAR(20),
    description TEXT,
    amount DECIMAL(18, 4) NOT NULL,
    currency VARCHAR(10) DEFAULT 'USD',
    record_hash VARCHAR(64) UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_div_account_date ON portfolio_dividends(account_id, date);
CREATE INDEX IF NOT EXISTS idx_div_symbol ON portfolio_dividends(symbol);

-- Interest
CREATE TABLE IF NOT EXISTS portfolio_interest (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    description TEXT,
    amount DECIMAL(18, 4) NOT NULL,
    currency VARCHAR(10) DEFAULT 'USD',
    record_hash VARCHAR(64) UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_int_account_date ON portfolio_interest(account_id, date);

-- Withholding Tax
CREATE TABLE IF NOT EXISTS portfolio_withholding_tax (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    symbol VARCHAR(20),
    description TEXT,
    amount DECIMAL(18, 4) NOT NULL,
    currency VARCHAR(10) DEFAULT 'USD',
    record_hash VARCHAR(64) UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_wht_account_date ON portfolio_withholding_tax(account_id, date);

-- Trades
CREATE TABLE IF NOT EXISTS portfolio_trades (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    description TEXT,
    quantity DECIMAL(18, 6) NOT NULL,
    price DECIMAL(18, 6) NOT NULL,
    proceeds DECIMAL(18, 4),
    commission DECIMAL(18, 4),
    fees DECIMAL(18, 4),
    realized_pnl DECIMAL(18, 4),
    asset_class VARCHAR(50),
    trade_type VARCHAR(20),
    currency VARCHAR(10) DEFAULT 'USD',
    record_hash VARCHAR(64) UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_trade_account_date ON portfolio_trades(account_id, date);
CREATE INDEX IF NOT EXISTS idx_trade_symbol ON portfolio_trades(symbol);

-- Fees
CREATE TABLE IF NOT EXISTS portfolio_fees (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    description TEXT,
    amount DECIMAL(18, 4) NOT NULL,
    fee_type VARCHAR(50),
    currency VARCHAR(10) DEFAULT 'USD',
    record_hash VARCHAR(64) UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_fee_account_date ON portfolio_fees(account_id, date);

-- NAV History
CREATE TABLE IF NOT EXISTS portfolio_nav_history (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    nav_total DECIMAL(18, 4),
    nav_cash DECIMAL(18, 4),
    nav_stock DECIMAL(18, 4),
    nav_bonds DECIMAL(18, 4),
    nav_options DECIMAL(18, 4),
    twr_percent DECIMAL(10, 6),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(account_id, date)
);
CREATE INDEX IF NOT EXISTS idx_nav_account_date ON portfolio_nav_history(account_id, date DESC);

-- Positions
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    description TEXT,
    quantity DECIMAL(18, 6) NOT NULL,
    cost_basis DECIMAL(18, 4),
    cost_price DECIMAL(18, 6),
    current_price DECIMAL(18, 6),
    market_value DECIMAL(18, 4),
    unrealized_pnl DECIMAL(18, 4),
    unrealized_pnl_pct DECIMAL(10, 4),
    asset_class VARCHAR(50),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(account_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_pos_account ON portfolio_positions(account_id);

-- Import History
CREATE TABLE IF NOT EXISTS portfolio_import_history (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(50) NOT NULL,
    filename VARCHAR(255),
    file_hash VARCHAR(64),
    period_start DATE,
    period_end DATE,
    records_imported INTEGER,
    imported_at TIMESTAMPTZ DEFAULT NOW()
);
"""


# =============================================================================
# REPOSITORY CLASS
# =============================================================================

class PortfolioRepository:
    """Repository for portfolio database operations."""

    def __init__(self):
        self.engine = get_engine()

    def init_schema(self):
        """Create database tables."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
        logger.info("Portfolio schema initialized")

    def _hash_record(self, *args) -> str:
        """Generate hash for deduplication."""
        data = "|".join(str(a) for a in args if a is not None)
        return hashlib.sha256(data.encode()).hexdigest()

    # =========================================================================
    # ACCOUNTS
    # =========================================================================

    def upsert_account(self, account_id: str, name: str = None,
                       account_type: str = None, base_currency: str = "USD",
                       is_consolidated: bool = False) -> int:
        """Insert or update account."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO portfolio_accounts 
                        (account_id, name, account_type, base_currency, is_consolidated, updated_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (account_id) DO UPDATE SET
                        name = COALESCE(EXCLUDED.name, portfolio_accounts.name),
                        account_type = COALESCE(EXCLUDED.account_type, portfolio_accounts.account_type),
                        is_consolidated = EXCLUDED.is_consolidated,
                        updated_at = NOW()
                    RETURNING id
                """, (account_id, name, account_type, base_currency, is_consolidated))
                result = cur.fetchone()
                return result[0] if result else None

    def get_account(self, account_id: str) -> Optional[Dict]:
        """Get account info."""
        query = "SELECT * FROM portfolio_accounts WHERE account_id = %(account_id)s"
        df = pd.read_sql(query, self.engine, params={"account_id": account_id})
        return df.iloc[0].to_dict() if not df.empty else None

    def get_all_accounts(self) -> List[Dict]:
        """Get all accounts."""
        df = pd.read_sql("SELECT * FROM portfolio_accounts ORDER BY account_id", self.engine)
        return df.to_dict('records')

    # =========================================================================
    # DEPOSITS & WITHDRAWALS
    # =========================================================================

    def insert_deposits_withdrawals(self, account_id: str, records: List[Dict]) -> int:
        """Insert deposit/withdrawal records with deduplication."""
        if not records:
            return 0

        inserted = 0
        with get_connection() as conn:
            with conn.cursor() as cur:
                for rec in records:
                    record_hash = self._hash_record(
                        account_id, rec.get('date'), rec.get('description'), rec.get('amount')
                    )
                    cur.execute("""
                        INSERT INTO portfolio_deposits_withdrawals 
                            (account_id, date, description, amount, currency, record_hash)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (record_hash) DO NOTHING
                    """, (
                        account_id, rec.get('date'), rec.get('description', ''),
                        rec.get('amount', 0), rec.get('currency', 'USD'), record_hash
                    ))
                    if cur.rowcount > 0:
                        inserted += 1
        return inserted

    def get_deposits_withdrawals(self, account_id: str) -> pd.DataFrame:
        """Get deposits/withdrawals for account."""
        query = """
            SELECT * FROM portfolio_deposits_withdrawals 
            WHERE account_id = %(account_id)s ORDER BY date
        """
        return pd.read_sql(query, self.engine, params={"account_id": account_id})

    def get_total_deposits(self, account_id: str) -> float:
        """Get net deposits for account."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COALESCE(SUM(amount), 0) 
                    FROM portfolio_deposits_withdrawals WHERE account_id = %s
                """, (account_id,))
                return float(cur.fetchone()[0])

    # =========================================================================
    # DIVIDENDS
    # =========================================================================

    def insert_dividends(self, account_id: str, records: List[Dict]) -> int:
        """Insert dividend records with deduplication."""
        if not records:
            return 0

        inserted = 0
        with get_connection() as conn:
            with conn.cursor() as cur:
                for rec in records:
                    record_hash = self._hash_record(
                        account_id, rec.get('date'), rec.get('symbol'),
                        rec.get('description'), rec.get('amount')
                    )
                    cur.execute("""
                        INSERT INTO portfolio_dividends 
                            (account_id, date, symbol, description, amount, currency, record_hash)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (record_hash) DO NOTHING
                    """, (
                        account_id, rec.get('date'), rec.get('symbol', ''),
                        rec.get('description', ''), rec.get('amount', 0),
                        rec.get('currency', 'USD'), record_hash
                    ))
                    if cur.rowcount > 0:
                        inserted += 1
        return inserted

    def get_dividends(self, account_id: str) -> pd.DataFrame:
        """Get dividends for account."""
        query = "SELECT * FROM portfolio_dividends WHERE account_id = %(account_id)s ORDER BY date"
        return pd.read_sql(query, self.engine, params={"account_id": account_id})

    def get_total_dividends(self, account_id: str) -> float:
        """Get total dividends for account."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COALESCE(SUM(amount), 0) 
                    FROM portfolio_dividends WHERE account_id = %s
                """, (account_id,))
                return float(cur.fetchone()[0])

    def get_dividends_by_symbol(self, account_id: str) -> Dict[str, float]:
        """Get dividends grouped by symbol."""
        query = """
            SELECT symbol, SUM(amount) as total
            FROM portfolio_dividends 
            WHERE account_id = %(account_id)s AND symbol IS NOT NULL AND symbol != ''
            GROUP BY symbol ORDER BY total DESC
        """
        df = pd.read_sql(query, self.engine, params={"account_id": account_id})
        return dict(zip(df['symbol'], df['total'])) if not df.empty else {}

    # =========================================================================
    # INTEREST
    # =========================================================================

    def insert_interest(self, account_id: str, records: List[Dict]) -> int:
        """Insert interest records with deduplication."""
        if not records:
            return 0

        inserted = 0
        with get_connection() as conn:
            with conn.cursor() as cur:
                for rec in records:
                    record_hash = self._hash_record(
                        account_id, rec.get('date'), rec.get('description'), rec.get('amount')
                    )
                    cur.execute("""
                        INSERT INTO portfolio_interest 
                            (account_id, date, description, amount, currency, record_hash)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (record_hash) DO NOTHING
                    """, (
                        account_id, rec.get('date'), rec.get('description', ''),
                        rec.get('amount', 0), rec.get('currency', 'USD'), record_hash
                    ))
                    if cur.rowcount > 0:
                        inserted += 1
        return inserted

    def get_total_interest(self, account_id: str) -> float:
        """Get total interest for account."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COALESCE(SUM(amount), 0) 
                    FROM portfolio_interest WHERE account_id = %s
                """, (account_id,))
                return float(cur.fetchone()[0])

    # =========================================================================
    # WITHHOLDING TAX
    # =========================================================================

    def insert_withholding_tax(self, account_id: str, records: List[Dict]) -> int:
        """Insert withholding tax records."""
        if not records:
            return 0

        inserted = 0
        with get_connection() as conn:
            with conn.cursor() as cur:
                for rec in records:
                    record_hash = self._hash_record(
                        account_id, rec.get('date'), rec.get('symbol'),
                        rec.get('description'), rec.get('amount')
                    )
                    cur.execute("""
                        INSERT INTO portfolio_withholding_tax 
                            (account_id, date, symbol, description, amount, currency, record_hash)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (record_hash) DO NOTHING
                    """, (
                        account_id, rec.get('date'), rec.get('symbol', ''),
                        rec.get('description', ''), rec.get('amount', 0),
                        rec.get('currency', 'USD'), record_hash
                    ))
                    if cur.rowcount > 0:
                        inserted += 1
        return inserted

    def get_total_withholding_tax(self, account_id: str) -> float:
        """Get total withholding tax for account."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COALESCE(SUM(amount), 0) 
                    FROM portfolio_withholding_tax WHERE account_id = %s
                """, (account_id,))
                return float(cur.fetchone()[0])

    # =========================================================================
    # TRADES
    # =========================================================================

    def insert_trades(self, account_id: str, records: List[Dict]) -> int:
        """Insert trade records."""
        if not records:
            return 0

        inserted = 0
        with get_connection() as conn:
            with conn.cursor() as cur:
                for rec in records:
                    record_hash = self._hash_record(
                        account_id, rec.get('date'), rec.get('symbol'),
                        rec.get('quantity'), rec.get('price'), rec.get('proceeds')
                    )
                    cur.execute("""
                        INSERT INTO portfolio_trades 
                            (account_id, date, symbol, description, quantity, price, 
                             proceeds, commission, fees, realized_pnl, asset_class, 
                             trade_type, currency, record_hash)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (record_hash) DO NOTHING
                    """, (
                        account_id, rec.get('date'), rec.get('symbol', ''),
                        rec.get('description', ''), rec.get('quantity', 0),
                        rec.get('price', 0), rec.get('proceeds', 0),
                        rec.get('commission', 0), rec.get('fees', 0),
                        rec.get('realized_pnl', 0), rec.get('asset_class', ''),
                        rec.get('trade_type', ''), rec.get('currency', 'USD'), record_hash
                    ))
                    if cur.rowcount > 0:
                        inserted += 1
        return inserted

    def get_trades(self, account_id: str, symbol: str = None) -> pd.DataFrame:
        """Get trades for account."""
        query = "SELECT * FROM portfolio_trades WHERE account_id = %(account_id)s"
        params = {"account_id": account_id}
        if symbol:
            query += " AND symbol = %(symbol)s"
            params["symbol"] = symbol
        query += " ORDER BY date DESC"
        return pd.read_sql(query, self.engine, params=params)

    # =========================================================================
    # FEES
    # =========================================================================

    def insert_fees(self, account_id: str, records: List[Dict]) -> int:
        """Insert fee records."""
        if not records:
            return 0

        inserted = 0
        with get_connection() as conn:
            with conn.cursor() as cur:
                for rec in records:
                    record_hash = self._hash_record(
                        account_id, rec.get('date'), rec.get('description'), rec.get('amount')
                    )
                    cur.execute("""
                        INSERT INTO portfolio_fees 
                            (account_id, date, description, amount, fee_type, currency, record_hash)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (record_hash) DO NOTHING
                    """, (
                        account_id, rec.get('date'), rec.get('description', ''),
                        rec.get('amount', 0), rec.get('fee_type', ''),
                        rec.get('currency', 'USD'), record_hash
                    ))
                    if cur.rowcount > 0:
                        inserted += 1
        return inserted

    def get_total_fees(self, account_id: str) -> float:
        """Get total fees for account."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COALESCE(SUM(amount), 0) 
                    FROM portfolio_fees WHERE account_id = %s
                """, (account_id,))
                return float(cur.fetchone()[0])

    # =========================================================================
    # NAV HISTORY
    # =========================================================================

    def upsert_nav(self, account_id: str, nav_date: date, nav_total: float,
                   nav_cash: float = 0, nav_stock: float = 0,
                   nav_bonds: float = 0, nav_options: float = 0,
                   twr_percent: float = None):
        """Insert or update NAV for a date."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO portfolio_nav_history 
                        (account_id, date, nav_total, nav_cash, nav_stock, nav_bonds, nav_options, twr_percent)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (account_id, date) DO UPDATE SET
                        nav_total = EXCLUDED.nav_total,
                        nav_cash = EXCLUDED.nav_cash,
                        nav_stock = EXCLUDED.nav_stock,
                        nav_bonds = EXCLUDED.nav_bonds,
                        nav_options = EXCLUDED.nav_options,
                        twr_percent = COALESCE(EXCLUDED.twr_percent, portfolio_nav_history.twr_percent)
                """, (account_id, nav_date, nav_total, nav_cash, nav_stock, nav_bonds, nav_options, twr_percent))

    def get_latest_nav(self, account_id: str) -> Optional[Dict]:
        """Get most recent NAV."""
        query = """
            SELECT * FROM portfolio_nav_history 
            WHERE account_id = %(account_id)s ORDER BY date DESC LIMIT 1
        """
        df = pd.read_sql(query, self.engine, params={"account_id": account_id})
        return df.iloc[0].to_dict() if not df.empty else None

    def get_nav_history(self, account_id: str) -> pd.DataFrame:
        """Get NAV history."""
        query = "SELECT * FROM portfolio_nav_history WHERE account_id = %(account_id)s ORDER BY date"
        return pd.read_sql(query, self.engine, params={"account_id": account_id})

    # =========================================================================
    # POSITIONS
    # =========================================================================

    def upsert_positions(self, account_id: str, positions: List[Dict]) -> int:
        """Insert or update positions."""
        if not positions:
            return 0

        updated = 0
        with get_connection() as conn:
            with conn.cursor() as cur:
                for pos in positions:
                    cur.execute("""
                        INSERT INTO portfolio_positions 
                            (account_id, symbol, description, quantity, cost_basis, cost_price,
                             current_price, market_value, unrealized_pnl, unrealized_pnl_pct, 
                             asset_class, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (account_id, symbol) DO UPDATE SET
                            quantity = EXCLUDED.quantity,
                            cost_basis = EXCLUDED.cost_basis,
                            cost_price = EXCLUDED.cost_price,
                            current_price = EXCLUDED.current_price,
                            market_value = EXCLUDED.market_value,
                            unrealized_pnl = EXCLUDED.unrealized_pnl,
                            unrealized_pnl_pct = EXCLUDED.unrealized_pnl_pct,
                            updated_at = NOW()
                    """, (
                        account_id, pos.get('symbol', ''), pos.get('description', ''),
                        pos.get('quantity', 0), pos.get('cost_basis', 0), pos.get('cost_price', 0),
                        pos.get('current_price', 0), pos.get('market_value', 0),
                        pos.get('unrealized_pnl', 0), pos.get('unrealized_pnl_pct', 0),
                        pos.get('asset_class', '')
                    ))
                    updated += 1
        return updated

    def get_positions(self, account_id: str) -> pd.DataFrame:
        """Get current positions."""
        query = """
            SELECT * FROM portfolio_positions 
            WHERE account_id = %(account_id)s ORDER BY market_value DESC
        """
        return pd.read_sql(query, self.engine, params={"account_id": account_id})

    def delete_positions(self, account_id: str) -> int:
        """Delete all positions for account."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM portfolio_positions WHERE account_id = %s", (account_id,))
                return cur.rowcount

    # =========================================================================
    # IMPORT STATEMENT
    # =========================================================================

    def import_statement(self, statement, filename: str = None) -> Dict[str, int]:
        """
        Import a parsed statement into the database.

        Args:
            statement: ParsedStatement object
            filename: Optional filename for tracking

        Returns:
            Dict with counts of inserted records
        """
        account_id = statement.account_info.account_id

        # Check if consolidated
        is_consolidated = "Consolidated" in account_id or "MULTI" in account_id.upper()
        clean_account_id = account_id.split(" (")[0].strip()

        # Upsert account
        self.upsert_account(
            account_id=clean_account_id,
            name=statement.account_info.name,
            account_type=statement.account_info.account_type,
            base_currency=statement.account_info.base_currency,
            is_consolidated=is_consolidated,
        )

        results = {'account_id': clean_account_id}

        # Helper to convert date
        def to_date(d):
            if d is None:
                return None
            if isinstance(d, datetime):
                return d.date()
            return d

        # Import deposits/withdrawals
        if statement.deposits_withdrawals:
            records = [{'date': to_date(dw.date), 'description': dw.description, 'amount': float(dw.amount or 0)}
                       for dw in statement.deposits_withdrawals if dw.amount]
            results['deposits_withdrawals'] = self.insert_deposits_withdrawals(clean_account_id, records)

        # Import dividends
        if statement.dividends:
            records = [{'date': to_date(d.date), 'symbol': d.symbol, 'description': d.description, 'amount': float(d.amount or 0)}
                       for d in statement.dividends if d.amount]
            results['dividends'] = self.insert_dividends(clean_account_id, records)

        # Import interest
        if statement.interest:
            records = [{'date': to_date(i.date), 'description': i.description, 'amount': float(i.amount or 0)}
                       for i in statement.interest if i.amount]
            results['interest'] = self.insert_interest(clean_account_id, records)

        # Import withholding tax
        if statement.withholding_tax:
            records = [{'date': to_date(w.date), 'symbol': getattr(w, 'symbol', ''), 'description': w.description, 'amount': float(w.amount or 0)}
                       for w in statement.withholding_tax if w.amount]
            results['withholding_tax'] = self.insert_withholding_tax(clean_account_id, records)

        # Import trades
        if statement.trades:
            records = [{
                'date': to_date(t.date), 'symbol': t.symbol, 'description': t.description,
                'quantity': float(t.quantity or 0), 'price': float(t.price or 0),
                'proceeds': float(t.proceeds or 0), 'commission': float(t.commission or 0),
                'realized_pnl': float(t.realized_pnl or 0), 'asset_class': t.asset_class,
                'trade_type': 'BUY' if (t.quantity or 0) > 0 else 'SELL'
            } for t in statement.trades]
            results['trades'] = self.insert_trades(clean_account_id, records)

        # Import fees
        if statement.fees:
            records = [{'date': to_date(f.date), 'description': f.description, 'amount': float(f.amount or 0)}
                       for f in statement.fees if f.amount]
            results['fees'] = self.insert_fees(clean_account_id, records)

        # Import positions
        if statement.open_positions:
            self.delete_positions(clean_account_id)
            records = [{
                'symbol': p.symbol, 'description': p.description, 'quantity': float(p.quantity or 0),
                'cost_basis': float(p.cost_basis or 0), 'cost_price': float(p.cost_price or 0),
                'current_price': float(p.current_price or 0), 'market_value': float(p.market_value or 0),
                'unrealized_pnl': float(p.unrealized_pnl or 0), 'unrealized_pnl_pct': float(p.unrealized_pnl_pct or 0),
                'asset_class': p.asset_class
            } for p in statement.open_positions]
            results['positions'] = self.upsert_positions(clean_account_id, records)

        # Save NAV
        nav = statement.nav
        if nav and nav.total:
            nav_date = date.today()
            if statement.period_end:
                try:
                    for fmt in ("%Y-%m-%d", "%B %d, %Y"):
                        try:
                            nav_date = datetime.strptime(str(statement.period_end), fmt).date()
                            break
                        except:
                            pass
                except:
                    pass

            twr = None
            if nav.twr_percent:
                try:
                    twr = float(str(nav.twr_percent).replace('%', ''))
                except:
                    pass

            self.upsert_nav(clean_account_id, nav_date, float(nav.total or 0),
                           float(nav.cash or 0), float(nav.stock or 0),
                           float(nav.bonds or 0), float(nav.options or 0), twr)

        # Record import
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO portfolio_import_history 
                        (account_id, filename, period_start, period_end, records_imported)
                    VALUES (%s, %s, %s, %s, %s)
                """, (clean_account_id, filename, statement.period_start, statement.period_end,
                      sum(v for k, v in results.items() if k != 'account_id' and isinstance(v, int))))

        return results

    # =========================================================================
    # ACCOUNT SUMMARY
    # =========================================================================

    def get_account_summary(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get complete account summary."""
        account = self.get_account(account_id)
        if not account:
            return None

        latest_nav = self.get_latest_nav(account_id)
        total_deposits = self.get_total_deposits(account_id)
        total_dividends = self.get_total_dividends(account_id)
        total_interest = self.get_total_interest(account_id)
        total_wht = self.get_total_withholding_tax(account_id)
        total_fees = self.get_total_fees(account_id)

        nav_total = float(latest_nav['nav_total']) if latest_nav else 0
        true_profit = nav_total - total_deposits
        true_profit_pct = (true_profit / abs(total_deposits) * 100) if total_deposits != 0 else 0

        return {
            'account': account,
            'nav': latest_nav,
            'total_deposits': total_deposits,
            'total_dividends': total_dividends,
            'total_interest': total_interest,
            'total_withholding_tax': total_wht,
            'total_fees': total_fees,
            'true_profit': true_profit,
            'true_profit_pct': true_profit_pct,
            'net_income': total_dividends + total_interest + total_wht,
        }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  Portfolio Database")
    print("=" * 60)

    repo = PortfolioRepository()

    if len(sys.argv) > 1 and sys.argv[1] == "init":
        print("\nInitializing schema...")
        repo.init_schema()
        print("✓ Done")

    elif len(sys.argv) > 2 and sys.argv[1] == "import":
        csv_path = sys.argv[2]
        print(f"\nImporting: {csv_path}")

        sys.path.insert(0, 'dashboard')
        from portfolio_tab import IBKRStatementParser

        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()

        parser = IBKRStatementParser()
        statement = parser.parse(content)

        print(f"Account: {statement.account_info.account_id}")
        print(f"NAV: ${statement.nav.total:,.2f}")

        repo.init_schema()
        results = repo.import_statement(statement, csv_path)

        print("\n✓ Import complete:")
        for k, v in results.items():
            if k != 'account_id' and isinstance(v, int):
                print(f"  {k}: {v}")

    elif len(sys.argv) > 1 and sys.argv[1] == "summary":
        account_id = sys.argv[2] if len(sys.argv) > 2 else None

        if account_id:
            summary = repo.get_account_summary(account_id)
            if summary:
                print(f"\nAccount: {account_id}")
                print(f"  NAV: ${summary['nav']['nav_total']:,.2f}" if summary['nav'] else "  NAV: N/A")
                print(f"  Deposits: ${summary['total_deposits']:,.2f}")
                print(f"  True Profit: ${summary['true_profit']:,.2f} ({summary['true_profit_pct']:.2f}%)")
                print(f"  Dividends: ${summary['total_dividends']:,.2f}")
                print(f"  Interest: ${summary['total_interest']:,.2f}")
            else:
                print(f"Account {account_id} not found")
        else:
            accounts = repo.get_all_accounts()
            print(f"\nAccounts: {len(accounts)}")
            for acc in accounts:
                print(f"  - {acc['account_id']}: {acc['name']}")

    else:
        print("""
Usage:
  python portfolio_db.py init                  # Create tables
  python portfolio_db.py import <csv_file>    # Import CSV
  python portfolio_db.py summary [account_id] # Show summary
        """)