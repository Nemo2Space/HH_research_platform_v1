

import pandas as pd
from datetime import date, datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from .connection import get_connection, get_engine


class Repository:
    """
    Database repository for all Alpha Platform data operations.
    """

    def __init__(self):
        self.engine = get_engine()

    # repository.py — ADD this FULL method (new) inside class Repository

    # repository.py — ADD this FULL method inside class Repository

    def get_latest_fundamentals_map(self, tickers: List[str]) -> pd.DataFrame:
        """
        Latest fundamentals per ticker (sector + dividend_yield + market_cap if available).
        Safe: returns empty df on any failure.
        """
        import pandas as pd

        if not tickers:
            return pd.DataFrame(columns=["ticker", "sector", "industry", "dividend_yield", "market_cap"])

        # normalize + unique
        norm = []
        for t in tickers:
            if t is None:
                continue
            s = str(t).strip().upper()
            if s:
                norm.append(s)
        tickers = list(dict.fromkeys(norm))
        if not tickers:
            return pd.DataFrame(columns=["ticker", "sector", "industry", "dividend_yield", "market_cap"])

        try:
            query = """
                SELECT DISTINCT ON (f.ticker)
                    f.ticker,
                    f.sector,
                    NULL::text AS industry,
                    f.dividend_yield,
                    f.market_cap
                FROM fundamentals f
                WHERE f.ticker = ANY(%(tickers)s)
                ORDER BY f.ticker, f.date DESC
            """
            df = pd.read_sql(query, self.engine, params={"tickers": tickers})
            if "ticker" in df.columns:
                df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
            return df
        except Exception:
            return pd.DataFrame(columns=["ticker", "sector", "industry", "dividend_yield", "market_cap"])

    # =========================================================================
    # UNIVERSE
    # =========================================================================


    def get_universe(self) -> List[str]:
        """Get list of all tickers in universe from config."""
        import os
        import csv

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config", "universe.csv"
        )

        tickers = []
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tickers.append(row['ticker'])

        return tickers

    # =========================================================================
    # PRICES
    # =========================================================================

    def save_prices(self, ticker: str, df: pd.DataFrame):
        """
        Save price data for a ticker.

        Args:
            ticker: Stock ticker
            df: DataFrame with columns: date, open, high, low, close, volume
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                for _, row in df.iterrows():
                    cur.execute("""
                        INSERT INTO prices (ticker, date, open, high, low, close, adj_close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, date) DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            adj_close = EXCLUDED.adj_close,
                            volume = EXCLUDED.volume
                    """, (
                        ticker,
                        row.get('date') or row.name,
                        row.get('open') or row.get('Open'),
                        row.get('high') or row.get('High'),
                        row.get('low') or row.get('Low'),
                        row.get('close') or row.get('Close'),
                        row.get('adj_close') or row.get('Adj Close') or row.get('close') or row.get('Close'),
                        row.get('volume') or row.get('Volume', 0),
                    ))

    def get_prices(self, ticker: str, start_date: Optional[date] = None,
                   end_date: Optional[date] = None) -> pd.DataFrame:
        """Get price history for a ticker."""
        query = "SELECT * FROM prices WHERE ticker = %(ticker)s"
        params = {"ticker": ticker}

        if start_date:
            query += " AND date >= %(start_date)s"
            params["start_date"] = start_date
        if end_date:
            query += " AND date <= %(end_date)s"
            params["end_date"] = end_date

        query += " ORDER BY date"

        return pd.read_sql(query, self.engine, params=params)

    def get_latest_price(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get most recent price for a ticker."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM prices 
                    WHERE ticker = %s 
                    ORDER BY date DESC 
                    LIMIT 1
                """, (ticker,))
                row = cur.fetchone()
                if row:
                    cols = [desc[0] for desc in cur.description]
                    return dict(zip(cols, row))
        return None

    # =========================================================================
    # SCREENER SCORES
    # =========================================================================

    def save_screener_score(self, ticker: str, score_date: date, scores: Dict[str, Any]):
        """
        Save screener scores for a ticker.

        Args:
            ticker: Stock ticker
            score_date: Date of the scores
            scores: Dict with score fields
        """

        # Convert numpy types to Python native types
        def to_python(val):
            if val is None:
                return None
            try:
                # Handle numpy types
                import numpy as np
                if isinstance(val, (np.integer, np.int64, np.int32)):
                    return int(val)
                if isinstance(val, (np.floating, np.float64, np.float32)):
                    return float(val)
                if isinstance(val, np.ndarray):
                    return val.tolist()
            except (ImportError, TypeError):
                pass
            return val

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                            INSERT INTO screener_scores (ticker, date, sentiment_score, sentiment_weighted,
                                                         fundamental_score, growth_score, dividend_score,
                                                         technical_score, gap_score, gap_type, likelihood_score,
                                                         analyst_positivity, target_upside_pct,
                                                         insider_signal, institutional_signal,
                                                         composite_score, total_score, article_count)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                    %s) ON CONFLICT (ticker, date) DO
                            UPDATE SET
                                sentiment_score = EXCLUDED.sentiment_score,
                                sentiment_weighted = EXCLUDED.sentiment_weighted,
                                fundamental_score = EXCLUDED.fundamental_score,
                                growth_score = EXCLUDED.growth_score,
                                dividend_score = EXCLUDED.dividend_score,
                                technical_score = EXCLUDED.technical_score,
                                gap_score = EXCLUDED.gap_score,
                                gap_type = EXCLUDED.gap_type,
                                likelihood_score = EXCLUDED.likelihood_score,
                                analyst_positivity = EXCLUDED.analyst_positivity,
                                target_upside_pct = EXCLUDED.target_upside_pct,
                                insider_signal = EXCLUDED.insider_signal,
                                institutional_signal = EXCLUDED.institutional_signal,
                                composite_score = EXCLUDED.composite_score,
                                total_score = EXCLUDED.total_score,
                                article_count = EXCLUDED.article_count
                            """, (
                                ticker, score_date,
                                to_python(scores.get('sentiment_score')),
                                to_python(scores.get('sentiment_weighted')),
                                to_python(scores.get('fundamental_score')),
                                to_python(scores.get('growth_score')),
                                to_python(scores.get('dividend_score')),
                                to_python(scores.get('technical_score')),
                                to_python(scores.get('gap_score')),
                                scores.get('gap_type'),
                                to_python(scores.get('likelihood_score')),
                                to_python(scores.get('analyst_positivity')),
                                to_python(scores.get('target_upside_pct')),
                                to_python(scores.get('insider_signal')),
                                to_python(scores.get('institutional_signal')),
                                to_python(scores.get('composite_score')),
                                to_python(scores.get('total_score')),
                                to_python(scores.get('article_count')),
                            ))


    def get_latest_scores(self, limit: int = 100) -> pd.DataFrame:
        """Get latest screener scores for all tickers."""
        query = """
            SELECT DISTINCT ON (ticker) *
            FROM screener_scores
            ORDER BY ticker, date DESC
            LIMIT %(limit)s
        """
        return pd.read_sql(query, self.engine, params={"limit": limit})

    # repository.py — REPLACE this FULL method inside class Repository

    def get_latest_scores(self, limit: int = 5000) -> pd.DataFrame:
        """
        Returns the latest screener_scores per ticker, enriched with sector (from fundamentals)
        and latest price. This matches what signals_tab expects (sector populated).
        """
        import pandas as pd

        try:
            sql = """
            WITH latest_scores AS (
                SELECT DISTINCT ON (s.ticker)
                    s.*
                FROM screener_scores s
                ORDER BY s.ticker, s.date DESC
            ),
            latest_fundamentals AS (
                SELECT DISTINCT ON (f.ticker)
                    f.ticker,
                    f.sector,
                    f.industry,
                    f.dividend_yield,
                    f.market_cap
                FROM fundamentals f
                ORDER BY f.ticker, f.date DESC NULLS LAST
            ),
            latest_prices AS (
                SELECT DISTINCT ON (p.ticker)
                    p.ticker,
                    p.price,
                    p.date AS price_date
                FROM prices p
                ORDER BY p.ticker, p.date DESC
            )
            SELECT
                ls.*,
                lf.sector AS sector,
                lf.industry AS industry,
                lf.dividend_yield AS dividend_yield,
                lf.market_cap AS market_cap,
                lp.price AS price,
                lp.price_date AS price_date
            FROM latest_scores ls
            LEFT JOIN latest_fundamentals lf ON lf.ticker = ls.ticker
            LEFT JOIN latest_prices lp ON lp.ticker = ls.ticker
            ORDER BY ls.total_score DESC NULLS LAST
            LIMIT %(limit)s
            """
            df = pd.read_sql(sql, self.engine, params={"limit": int(limit)})
            if "ticker" in df.columns:
                df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
            return df
        except Exception:
            # Fallback: old behavior (no sector), so UI still works; diagnostics will show sector missing.
            try:
                sql_fallback = """
                WITH latest_scores AS (
                    SELECT DISTINCT ON (s.ticker)
                        s.*
                    FROM screener_scores s
                    ORDER BY s.ticker, s.date DESC
                )
                SELECT *
                FROM latest_scores
                ORDER BY total_score DESC NULLS LAST
                LIMIT %(limit)s
                """
                df = pd.read_sql(sql_fallback, self.engine, params={"limit": int(limit)})
                if "ticker" in df.columns:
                    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
                return df
            except Exception:
                return pd.DataFrame()

    def get_latest_fundamentals_sector_map(self, tickers: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Returns {TICKER: sector} using fundamentals.
        IMPORTANT: prefers the most recent NON-NULL sector (prevents NULL-overwrite).
        """
        if tickers:
            tickers = [t.strip().upper() for t in tickers if t and str(t).strip()]
            if not tickers:
                return {}

            query = """
                SELECT DISTINCT ON (f.ticker)
                    UPPER(f.ticker) AS ticker,
                    f.sector
                FROM fundamentals f
                WHERE f.ticker = ANY(%(tickers)s)
                ORDER BY
                    f.ticker,
                    (f.sector IS NOT NULL) DESC,
                    f.date DESC NULLS LAST
            """
            df = pd.read_sql(query, self.engine, params={"tickers": tickers})
        else:
            query = """
                SELECT DISTINCT ON (f.ticker)
                    UPPER(f.ticker) AS ticker,
                    f.sector
                FROM fundamentals f
                ORDER BY
                    f.ticker,
                    (f.sector IS NOT NULL) DESC,
                    f.date DESC NULLS LAST
            """
            df = pd.read_sql(query, self.engine)

        out: Dict[str, str] = {}
        if df is None or df.empty:
            return out

        for _, r in df.iterrows():
            t = (r.get("ticker") or "").strip().upper()
            s = r.get("sector")
            if not t:
                continue
            if s is None:
                continue
            s = str(s).strip()
            if not s:
                continue
            out[t] = s

        return out

    # =========================================================================
    # TRADING SIGNALS
    # =========================================================================

    def save_signal(self, ticker: str, signal_date: date, signal: Dict[str, Any]):
        """Save trading signal for a ticker."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trading_signals (
                        ticker, date, signal_type, signal_strength, signal_color,
                        signal_reason, sentiment_score, fundamental_score,
                        gap_score, likelihood_score, analyst_positivity
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, date) DO UPDATE SET
                        signal_type = EXCLUDED.signal_type,
                        signal_strength = EXCLUDED.signal_strength,
                        signal_color = EXCLUDED.signal_color,
                        signal_reason = EXCLUDED.signal_reason,
                        sentiment_score = EXCLUDED.sentiment_score,
                        fundamental_score = EXCLUDED.fundamental_score,
                        gap_score = EXCLUDED.gap_score,
                        likelihood_score = EXCLUDED.likelihood_score,
                        analyst_positivity = EXCLUDED.analyst_positivity
                """, (
                    ticker, signal_date,
                    signal.get('type'),
                    signal.get('strength'),
                    signal.get('color'),
                    signal.get('reasons', [''])[0] if signal.get('reasons') else signal.get('reason', ''),
                    signal.get('sentiment_score'),
                    signal.get('fundamental_score'),
                    signal.get('gap_score'),
                    signal.get('likelihood_score'),
                    signal.get('analyst_positivity'),
                ))

    # =========================================================================
    # PORTFOLIO SNAPSHOTS
    # =========================================================================

    def save_portfolio_snapshot(self, snapshot: Dict[str, Any]) -> int:
        """
        Save a portfolio snapshot.
        Uses UPSERT to update if same date/account exists.

        Returns:
            The ID of the created/updated entry
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                            INSERT INTO portfolio_snapshots (snapshot_date, account_id, net_liquidation,
                                                             total_cash,
                                                             gross_position_value, realized_pnl, unrealized_pnl,
                                                             daily_pnl, daily_return_pct, cumulative_return_pct,
                                                             benchmark_value, benchmark_return_pct,
                                                             alpha_vs_benchmark,
                                                             position_count, notes)
                            VALUES (%(snapshot_date)s, %(account_id)s, %(net_liquidation)s, %(total_cash)s,
                                    %(gross_position_value)s, %(realized_pnl)s, %(unrealized_pnl)s,
                                    %(daily_pnl)s, %(daily_return_pct)s, %(cumulative_return_pct)s,
                                    %(benchmark_value)s, %(benchmark_return_pct)s, %(alpha_vs_benchmark)s,
                                    %(position_count)s, %(notes)s) ON CONFLICT (snapshot_date, account_id) DO
                            UPDATE SET
                                net_liquidation = EXCLUDED.net_liquidation,
                                total_cash = EXCLUDED.total_cash,
                                gross_position_value = EXCLUDED.gross_position_value,
                                realized_pnl = EXCLUDED.realized_pnl,
                                unrealized_pnl = EXCLUDED.unrealized_pnl,
                                daily_pnl = EXCLUDED.daily_pnl,
                                daily_return_pct = EXCLUDED.daily_return_pct,
                                cumulative_return_pct = EXCLUDED.cumulative_return_pct,
                                benchmark_value = EXCLUDED.benchmark_value,
                                benchmark_return_pct = EXCLUDED.benchmark_return_pct,
                                alpha_vs_benchmark = EXCLUDED.alpha_vs_benchmark,
                                position_count = EXCLUDED.position_count,
                                notes = EXCLUDED.notes
                                RETURNING id
                            """, snapshot)
                result = cur.fetchone()
                return result[0] if result else None

    def get_portfolio_snapshots(self, days: int = 30, account_id: str = None) -> pd.DataFrame:
        """Get portfolio snapshots for the last N days."""
        query = """
                SELECT * \
                FROM portfolio_snapshots
                WHERE snapshot_date >= CURRENT_DATE - %(days)s \
                """
        params = {'days': days}

        if account_id:
            query += " AND account_id = %(account_id)s"
            params['account_id'] = account_id

        query += " ORDER BY snapshot_date ASC"

        return pd.read_sql(query, self.engine, params=params)

    def get_latest_snapshot(self, account_id: str = None) -> Optional[Dict[str, Any]]:
        """Get the most recent portfolio snapshot."""
        query = "SELECT * FROM portfolio_snapshots"
        params = {}

        if account_id:
            query += " WHERE account_id = %(account_id)s"
            params['account_id'] = account_id

        query += " ORDER BY snapshot_date DESC LIMIT 1"

        df = pd.read_sql(query, self.engine, params=params)
        if not df.empty:
            return df.iloc[0].to_dict()
        return None

    def get_first_snapshot(self, account_id: str = None) -> Optional[Dict[str, Any]]:
        """Get the first portfolio snapshot (for YTD calculations)."""
        query = "SELECT * FROM portfolio_snapshots"
        params = {}

        if account_id:
            query += " WHERE account_id = %(account_id)s"
            params['account_id'] = account_id

        query += " ORDER BY snapshot_date ASC LIMIT 1"

        df = pd.read_sql(query, self.engine, params=params)
        if not df.empty:
            return df.iloc[0].to_dict()
        return None

    def get_performance_summary(self, account_id: str = None) -> Dict[str, Any]:
        """Calculate performance summary from snapshots."""
        snapshots = self.get_portfolio_snapshots(days=365, account_id=account_id)

        if snapshots.empty:
            return {
                'total_snapshots': 0,
                'first_date': None,
                'last_date': None,
                'starting_value': 0,
                'current_value': 0,
                'total_return_pct': 0,
                'total_return_amt': 0,
                'avg_daily_return': 0,
                'best_day': 0,
                'worst_day': 0,
                'benchmark_return': 0,
                'alpha': 0
            }

        first = snapshots.iloc[0]
        last = snapshots.iloc[-1]

        starting_value = float(first['net_liquidation']) if first['net_liquidation'] else 0
        current_value = float(last['net_liquidation']) if last['net_liquidation'] else 0

        total_return_pct = (
                    (current_value - starting_value) / starting_value * 100) if starting_value > 0 else 0

        return {
            'total_snapshots': len(snapshots),
            'first_date': first['snapshot_date'],
            'last_date': last['snapshot_date'],
            'starting_value': starting_value,
            'current_value': current_value,
            'total_return_pct': round(total_return_pct, 2),
            'total_return_amt': round(current_value - starting_value, 2),
            'avg_daily_return': round(snapshots['daily_return_pct'].mean(),
                                      4) if 'daily_return_pct' in snapshots else 0,
            'best_day': round(snapshots['daily_return_pct'].max(), 2) if 'daily_return_pct' in snapshots else 0,
            'worst_day': round(snapshots['daily_return_pct'].min(),
                               2) if 'daily_return_pct' in snapshots else 0,
            'benchmark_return': round(float(last['benchmark_return_pct']), 2) if last[
                'benchmark_return_pct'] else 0,
            'alpha': round(float(last['alpha_vs_benchmark']), 2) if last['alpha_vs_benchmark'] else 0
        }

        # =========================================================================
        # TRADE JOURNAL
        # =========================================================================

    def save_journal_entry(self, entry: Dict[str, Any]) -> int:
        """
        Save a new trade journal entry.

        Returns:
            The ID of the created entry
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                            INSERT INTO trade_journal (ticker, action, entry_date, entry_price, quantity,
                                                       thesis, target_price, stop_loss, time_horizon,
                                                       conviction, tags, status, notes)
                            VALUES (%(ticker)s, %(action)s, %(entry_date)s, %(entry_price)s, %(quantity)s,
                                    %(thesis)s, %(target_price)s, %(stop_loss)s, %(time_horizon)s,
                                    %(conviction)s, %(tags)s, %(status)s, %(notes)s) RETURNING id
                            """, {
                                'ticker': entry.get('ticker'),
                                'action': entry.get('action', 'BUY'),
                                'entry_date': entry.get('entry_date'),
                                'entry_price': entry.get('entry_price'),
                                'quantity': entry.get('quantity'),
                                'thesis': entry.get('thesis'),
                                'target_price': entry.get('target_price'),
                                'stop_loss': entry.get('stop_loss'),
                                'time_horizon': entry.get('time_horizon'),
                                'conviction': entry.get('conviction'),
                                'tags': entry.get('tags'),
                                'status': entry.get('status', 'open'),
                                'notes': entry.get('notes')
                            })
                result = cur.fetchone()
                return result[0] if result else None

    def update_journal_entry(self, entry_id: int, updates: Dict[str, Any]):
        """Update an existing journal entry."""
        # Build dynamic update query
        set_clauses = []
        params = {'id': entry_id}

        allowed_fields = [
            'thesis', 'target_price', 'stop_loss', 'time_horizon',
            'conviction', 'tags', 'status', 'exit_date', 'exit_price',
            'exit_reason', 'pnl_amount', 'pnl_percent', 'notes'
        ]

        for field in allowed_fields:
            if field in updates:
                set_clauses.append(f"{field} = %({field})s")
                params[field] = updates[field]

        if not set_clauses:
            return

        query = f"UPDATE trade_journal SET {', '.join(set_clauses)} WHERE id = %(id)s"

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)

    def close_journal_entry(self, entry_id: int, exit_price: float, exit_reason: str = None):
        """Close a journal entry with exit details."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Get entry details first
                cur.execute("SELECT entry_price, quantity FROM trade_journal WHERE id = %s", (entry_id,))
                row = cur.fetchone()

                if row:
                    entry_price, quantity = row
                    pnl_amount = (exit_price - float(entry_price)) * float(
                        quantity) if entry_price and quantity else None
                    pnl_percent = ((exit_price - float(entry_price)) / float(
                        entry_price) * 100) if entry_price else None

                    cur.execute("""
                                UPDATE trade_journal
                                SET status      = 'closed',
                                    exit_date   = CURRENT_DATE,
                                    exit_price  = %s,
                                    exit_reason = %s,
                                    pnl_amount  = %s,
                                    pnl_percent = %s
                                WHERE id = %s
                                """, (exit_price, exit_reason, pnl_amount, pnl_percent, entry_id))

    def get_journal_entries(self, ticker: str = None, status: str = None, limit: int = 50) -> pd.DataFrame:
        """Get journal entries with optional filters."""
        query = "SELECT * FROM trade_journal WHERE 1=1"
        params = {}

        if ticker:
            query += " AND ticker = %(ticker)s"
            params['ticker'] = ticker.upper()

        if status:
            query += " AND status = %(status)s"
            params['status'] = status

        query += " ORDER BY entry_date DESC, created_at DESC LIMIT %(limit)s"
        params['limit'] = limit

        return pd.read_sql(query, self.engine, params=params)

    def get_open_journal_entries(self) -> pd.DataFrame:
        """Get all open journal entries."""
        return self.get_journal_entries(status='open', limit=500)

    def get_journal_entry_for_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get the most recent open journal entry for a ticker."""
        query = """
                SELECT * \
                FROM trade_journal
                WHERE ticker = %(ticker)s \
                  AND status = 'open'
                ORDER BY entry_date DESC LIMIT 1 \
                """
        df = pd.read_sql(query, self.engine, params={'ticker': ticker.upper()})

        if not df.empty:
            return df.iloc[0].to_dict()
        return None

    def get_journal_summary(self) -> Dict[str, Any]:
        """Get summary statistics for trade journal."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Open positions
                cur.execute("SELECT COUNT(*) FROM trade_journal WHERE status = 'open'")
                open_count = cur.fetchone()[0]

                # Closed positions stats
                cur.execute("""
                            SELECT COUNT(*)                                          as total_closed,
                                   SUM(CASE WHEN pnl_percent > 0 THEN 1 ELSE 0 END)  as winners,
                                   SUM(CASE WHEN pnl_percent <= 0 THEN 1 ELSE 0 END) as losers,
                                   AVG(pnl_percent)                                  as avg_return,
                                   SUM(pnl_amount)                                   as total_pnl
                            FROM trade_journal
                            WHERE status = 'closed'
                            """)
                row = cur.fetchone()

                return {
                    'open_positions': open_count,
                    'closed_positions': row[0] or 0,
                    'winners': row[1] or 0,
                    'losers': row[2] or 0,
                    'win_rate': (row[1] / row[0] * 100) if row[0] and row[0] > 0 else 0,
                    'avg_return': float(row[3]) if row[3] else 0,
                    'total_pnl': float(row[4]) if row[4] else 0
                }

    def delete_journal_entry(self, entry_id: int) -> bool:
        """Delete a journal entry by ID."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM trade_journal WHERE id = %s", (entry_id,))
                return cur.rowcount > 0




    def get_latest_signals(self) -> pd.DataFrame:
        """Get latest signals for all tickers."""
        query = """
            SELECT DISTINCT ON (ticker) *
            FROM trading_signals
            ORDER BY ticker, date DESC
        """
        return pd.read_sql(query, self.engine)

    def get_signals_by_type(self, signal_types: List[str]) -> pd.DataFrame:
        """Get latest signals filtered by type."""
        query = """
            SELECT DISTINCT ON (ticker) *
            FROM trading_signals
            WHERE signal_type = ANY(%(signal_types)s)
            ORDER BY ticker, date DESC
        """
        return pd.read_sql(query, self.engine, params={"signal_types": signal_types})

    # =========================================================================
    # NEWS ARTICLES
    # =========================================================================

    def save_news_article(self, article: Dict[str, Any]):
        """Save a news article. Skips duplicates based on ticker+url."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                # First check if article already exists (by ticker + url)
                url = article.get('url', '')
                ticker = article.get('ticker', '')

                if url and ticker:
                    cur.execute("""
                        SELECT id FROM news_articles 
                        WHERE ticker = %s AND url = %s
                        LIMIT 1
                    """, (ticker, url))

                    if cur.fetchone():
                        # Article exists - update created_at to mark as "refreshed"
                        cur.execute("""
                            UPDATE news_articles SET created_at = NOW()
                            WHERE ticker = %s AND url = %s
                        """, (ticker, url))
                        return  # Already exists

                # Insert new article
                cur.execute("""
                    INSERT INTO news_articles (
                        ticker, headline, snippet, url, source, author,
                        published_at, credibility_score, ai_sentiment_fast, is_relevant, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """, (
                    ticker,
                    article.get('headline') or article.get('title'),
                    article.get('snippet') or article.get('description'),
                    url,
                    article.get('source'),
                    article.get('author'),
                    article.get('published_at') or article.get('publishedAt'),
                    article.get('credibility_score', 5),
                    article.get('ai_sentiment_fast'),
                    article.get('is_relevant', True),
                ))

    def get_news_for_ticker(self, ticker: str, days: int = 7) -> pd.DataFrame:
        """Get recent news for a ticker."""
        query = """
            SELECT * FROM news_articles
            WHERE ticker = %(ticker)s
              AND published_at >= NOW() - INTERVAL '%(days)s days'
            ORDER BY published_at DESC
        """
        return pd.read_sql(query, self.engine, params={"ticker": ticker, "days": days})

    # =========================================================================
    # SENTIMENT SCORES
    # =========================================================================

    def save_sentiment_score(self, ticker: str, score_date: date, scores: Dict[str, Any]):
        """Save aggregated sentiment score."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sentiment_scores (
                        ticker, date, sentiment_raw, sentiment_weighted,
                        ai_sentiment_fast, ai_sentiment_deep,
                        article_count, relevant_article_count, sentiment_class
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, date) DO UPDATE SET
                        sentiment_raw = EXCLUDED.sentiment_raw,
                        sentiment_weighted = EXCLUDED.sentiment_weighted,
                        ai_sentiment_fast = EXCLUDED.ai_sentiment_fast,
                        ai_sentiment_deep = EXCLUDED.ai_sentiment_deep,
                        article_count = EXCLUDED.article_count,
                        relevant_article_count = EXCLUDED.relevant_article_count,
                        sentiment_class = EXCLUDED.sentiment_class
                """, (
                    ticker, score_date,
                    scores.get('sentiment_raw'),
                    scores.get('sentiment_weighted'),
                    scores.get('ai_sentiment_fast'),
                    scores.get('ai_sentiment_deep'),
                    scores.get('article_count'),
                    scores.get('relevant_article_count'),
                    scores.get('sentiment_class'),
                ))

    # =========================================================================
    # COMMITTEE DECISIONS
    # =========================================================================

    def save_committee_decision(self, decision: Dict[str, Any]):
        """Save a committee decision."""
        import json

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO committee_decisions (
                        ticker, date, persona, verdict, conviction,
                        expected_alpha_bps, horizon_days, buy_prob, confidence,
                        risks, rationale, transcript_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, date, persona) DO UPDATE SET
                        verdict = EXCLUDED.verdict,
                        conviction = EXCLUDED.conviction,
                        expected_alpha_bps = EXCLUDED.expected_alpha_bps,
                        horizon_days = EXCLUDED.horizon_days,
                        buy_prob = EXCLUDED.buy_prob,
                        confidence = EXCLUDED.confidence,
                        risks = EXCLUDED.risks,
                        rationale = EXCLUDED.rationale,
                        transcript_id = EXCLUDED.transcript_id
                """, (
                    decision.get('ticker'),
                    decision.get('date'),
                    decision.get('persona', 'neutral'),
                    decision.get('verdict'),
                    decision.get('conviction'),
                    decision.get('expected_alpha_bps'),
                    decision.get('horizon_days'),
                    decision.get('buy_prob'),
                    decision.get('confidence'),
                    json.dumps(decision.get('risks', [])),
                    decision.get('rationale'),
                    decision.get('transcript_id'),
                ))

    def get_latest_decisions(self, persona: str = 'neutral') -> pd.DataFrame:
        """Get latest committee decisions."""
        query = """
            SELECT DISTINCT ON (ticker) *
            FROM committee_decisions
            WHERE persona = %(persona)s
            ORDER BY ticker, date DESC
        """
        return pd.read_sql(query, self.engine, params={"persona": persona})

    # =========================================================================
    # SYSTEM STATUS
    # =========================================================================

    def update_system_status(self, component: str, status: str,
                             progress_pct: Optional[int] = None,
                             progress_message: Optional[str] = None,
                             error_message: Optional[str] = None):
        """Update system component status."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                if status == 'running':
                    cur.execute("""
                        UPDATE system_status SET
                            status = %s,
                            last_run_at = NOW(),
                            progress_pct = %s,
                            progress_message = %s,
                            error_message = NULL,
                            updated_at = NOW()
                        WHERE component = %s
                    """, (status, progress_pct, progress_message, component))
                elif status == 'completed':
                    cur.execute("""
                        UPDATE system_status SET
                            status = 'stopped',
                            last_completed_at = NOW(),
                            progress_pct = 100,
                            progress_message = %s,
                            error_message = NULL,
                            updated_at = NOW()
                        WHERE component = %s
                    """, (progress_message, component))
                elif status == 'error':
                    cur.execute("""
                        UPDATE system_status SET
                            status = %s,
                            progress_message = NULL,
                            error_message = %s,
                            updated_at = NOW()
                        WHERE component = %s
                    """, (status, error_message, component))
                else:
                    cur.execute("""
                        UPDATE system_status SET
                            status = %s,
                            progress_pct = %s,
                            progress_message = %s,
                            updated_at = NOW()
                        WHERE component = %s
                    """, (status, progress_pct, progress_message, component))

    def get_system_status(self) -> pd.DataFrame:
        """Get status of all system components."""
        return pd.read_sql("SELECT * FROM system_status", self.engine)