"""
Signal Performance Tracker

Tracks how accurate trading signals are over time by measuring
actual returns after BUY/SELL signals are generated.

Author: Alpha Research Platform
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yfinance as yf

from src.db.connection import get_connection, get_engine
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SignalPerformance:
    """Performance metrics for a signal type."""
    signal_type: str
    total_signals: int
    win_rate_5d: float
    win_rate_10d: float
    win_rate_30d: float
    avg_return_5d: float
    avg_return_10d: float
    avg_return_30d: float
    best_ticker: str
    best_return: float
    worst_ticker: str
    worst_return: float


class SignalPerformanceTracker:
    """
    Track and analyze signal performance over time.

    Measures:
    - Win rate (% of signals that were profitable)
    - Average return after signal
    - Best/worst performing signals
    - Performance by signal type (BUY, SELL, STRONG BUY, etc.)
    """

    def __init__(self):
        self.engine = get_engine()

    def get_signals_with_returns(self, days_back: int = 90) -> pd.DataFrame:
        """
        Get historical signals with actual returns calculated.

        UPDATED: First tries cached returns from historical_scores (instant).
        Falls back to yfinance download only if cached data not available.

        Args:
            days_back: How many days of signals to analyze

        Returns:
            DataFrame with signals and their returns
        """
        # FIRST: Try cached returns from historical_scores (instant, no download)
        cached_query = """
            SELECT 
                ticker, 
                score_date as signal_date,
                signal_type,
                total_score as signal_strength,
                sentiment as sentiment_score,
                fundamental_score,
                return_1d as return_2d,
                return_5d,
                return_10d,
                return_20d as return_30d
            FROM historical_scores
            WHERE score_date >= CURRENT_DATE - INTERVAL '%s days'
              AND score_date <= CURRENT_DATE - INTERVAL '5 days'
              AND return_5d IS NOT NULL
            ORDER BY score_date DESC
        """

        try:
            df = pd.read_sql(cached_query % days_back, self.engine)

            if not df.empty:
                logger.info(f"Found {len(df)} signals with cached returns (no download needed)")
                return df
        except Exception as e:
            logger.debug(f"Cached returns not available: {e}")

        # FALLBACK: Original method - trading_signals + yfinance download
        query = """
            SELECT 
                ticker, 
                date as signal_date,
                signal_type,
                signal_strength,
                sentiment_score,
                fundamental_score
            FROM trading_signals
            WHERE date >= CURRENT_DATE - INTERVAL '%s days'
              AND date <= CURRENT_DATE - INTERVAL '5 days'  -- Need at least 5 days for returns
            ORDER BY date DESC
        """

        try:
            df = pd.read_sql(query % days_back, self.engine)

            if df.empty:
                logger.warning("No signals found for performance tracking")
                return df

            logger.info(f"Found {len(df)} signals to analyze (downloading prices...)")

            # Calculate returns for each signal
            df = self._calculate_returns(df)

            return df

        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            return pd.DataFrame()

    # Known problematic tickers that yfinance can't handle
    SKIP_TICKERS = {'SQ', 'TWTR', 'FB'}  # Renamed/delisted tickers

    def _fix_ticker(self, ticker: str) -> str:
        """Fix common ticker symbol issues for yfinance."""
        # BRK.B -> BRK-B (yfinance format)
        if '.' in ticker:
            ticker = ticker.replace('.', '-')
        return ticker

    def _fetch_prices_batch(self, tickers: list, start_date, end_date) -> pd.DataFrame:
        """Fetch prices for a batch of tickers with error handling."""
        if not tickers:
            return pd.DataFrame()

        # Filter out known problematic tickers
        tickers = [t for t in tickers if t not in self.SKIP_TICKERS]
        if not tickers:
            return pd.DataFrame()

        # Fix ticker symbols
        fixed_tickers = [self._fix_ticker(t) for t in tickers]
        ticker_map = dict(zip(fixed_tickers, tickers))  # Map back to original

        try:
            # Suppress yfinance warnings
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                data = yf.download(
                    fixed_tickers,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                    threads=True,
                    timeout=30
                )

            if data.empty:
                return pd.DataFrame()

            # Get Close prices
            if 'Close' in data.columns or (isinstance(data.columns, pd.MultiIndex) and 'Close' in data.columns.get_level_values(0)):
                prices = data['Close'] if 'Close' in data else data.xs('Close', axis=1, level=0)
            else:
                prices = data

            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=fixed_tickers[0])

            # Rename columns back to original ticker symbols
            prices.columns = [ticker_map.get(c, c) for c in prices.columns]

            return prices

        except Exception as e:
            logger.warning(f"Batch download error for {len(tickers)} tickers: {e}")
            return pd.DataFrame()

    def _calculate_returns(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate actual returns after each signal."""

        # Get unique tickers
        tickers = signals_df['ticker'].unique().tolist()

        # Fetch price data in batches
        logger.info(f"Fetching price data for {len(tickers)} tickers...")

        # Get date range - ensure all dates are date objects
        min_date = pd.to_datetime(signals_df['signal_date'].min()).date()
        max_date = pd.to_datetime(signals_df['signal_date'].max()).date() + timedelta(days=35)
        today = date.today()

        start_date = min_date - timedelta(days=5)
        end_date = min(max_date, today)

        # Download in batches of 20 to avoid timeouts
        BATCH_SIZE = 20
        all_prices = []

        for i in range(0, len(tickers), BATCH_SIZE):
            batch = tickers[i:i + BATCH_SIZE]
            logger.debug(f"Fetching batch {i//BATCH_SIZE + 1}/{(len(tickers)-1)//BATCH_SIZE + 1}: {len(batch)} tickers")

            batch_prices = self._fetch_prices_batch(batch, start_date, end_date)
            if not batch_prices.empty:
                all_prices.append(batch_prices)

        # Combine all batches
        if all_prices:
            price_data = pd.concat(all_prices, axis=1)
            # Remove duplicate columns if any
            price_data = price_data.loc[:, ~price_data.columns.duplicated()]
        else:
            logger.error("Failed to fetch any price data")
            signals_df['return_2d'] = None
            signals_df['return_5d'] = None
            signals_df['return_10d'] = None
            signals_df['return_30d'] = None
            return signals_df

        logger.info(f"Got prices for {len(price_data.columns)} tickers")

        # Ensure price index is timezone-naive for comparison
        if price_data.index.tz is not None:
            price_data.index = price_data.index.tz_localize(None)

        # Convert index to date for easier comparison
        price_dates = pd.Series(price_data.index.date, index=price_data.index)

        # Calculate returns for each signal
        returns_2d = []
        returns_5d = []
        returns_10d = []
        returns_30d = []

        matched = 0
        for _, row in signals_df.iterrows():
            ticker = row['ticker']
            # Convert signal_date to date object
            try:
                if isinstance(row['signal_date'], str):
                    signal_date = pd.to_datetime(row['signal_date']).date()
                elif hasattr(row['signal_date'], 'date'):
                    signal_date = row['signal_date'].date()
                else:
                    signal_date = row['signal_date']
            except:
                signal_date = row['signal_date']

            try:
                if ticker not in price_data.columns:
                    returns_2d.append(None)
                    returns_5d.append(None)
                    returns_10d.append(None)
                    returns_30d.append(None)
                    continue

                ticker_prices = price_data[ticker].dropna()

                if ticker_prices.empty:
                    returns_2d.append(None)
                    returns_5d.append(None)
                    returns_10d.append(None)
                    returns_30d.append(None)
                    continue
                    continue

                ticker_prices = price_data[ticker].dropna()

                if ticker_prices.empty:
                    returns_5d.append(None)
                    returns_10d.append(None)
                    returns_30d.append(None)
                    continue

                # Find prices on or after signal date
                ticker_dates = pd.Series(ticker_prices.index.date, index=ticker_prices.index)
                mask = ticker_dates >= signal_date
                future_prices = ticker_prices[mask]

                if len(future_prices) < 1:
                    returns_2d.append(None)
                    returns_5d.append(None)
                    returns_10d.append(None)
                    returns_30d.append(None)
                    continue

                signal_price = future_prices.iloc[0]
                matched += 1

                # Calculate returns at different horizons
                def get_return(days):
                    target_date = signal_date + timedelta(days=days)
                    future_mask = ticker_dates >= target_date
                    future = ticker_prices[future_mask]
                    if len(future) > 0:
                        return ((future.iloc[0] - signal_price) / signal_price) * 100
                    return None

                returns_2d.append(get_return(2))
                returns_5d.append(get_return(5))
                returns_10d.append(get_return(10))
                returns_30d.append(get_return(30))

            except Exception as e:
                logger.debug(f"Error calculating return for {ticker}: {e}")
                returns_2d.append(None)
                returns_5d.append(None)
                returns_10d.append(None)
                returns_30d.append(None)

        signals_df['return_2d'] = returns_2d
        signals_df['return_5d'] = returns_5d
        signals_df['return_10d'] = returns_10d
        signals_df['return_30d'] = returns_30d

        # Log match statistics
        has_2d = sum(1 for r in returns_2d if r is not None)
        has_5d = sum(1 for r in returns_5d if r is not None)
        has_10d = sum(1 for r in returns_10d if r is not None)
        logger.info(f"Matched {matched} signals, 2d returns: {has_2d}, 5d returns: {has_5d}, 10d returns: {has_10d}")

        return signals_df

    def get_performance_by_signal_type(self, days_back: int = 90) -> Dict[str, SignalPerformance]:
        """
        Get performance metrics grouped by signal type.

        Returns:
            Dict of signal_type -> SignalPerformance
        """
        df = self.get_signals_with_returns(days_back)

        if df.empty:
            return {}

        results = {}

        for signal_type in df['signal_type'].unique():
            if pd.isna(signal_type):
                continue

            type_df = df[df['signal_type'] == signal_type].copy()

            # Calculate win rates (positive return = win)
            # For SELL signals, negative return = win
            is_sell = 'SELL' in str(signal_type).upper()

            def calc_win_rate(returns_col):
                valid = type_df[returns_col].dropna()
                if len(valid) == 0:
                    return 0.0
                if is_sell:
                    return (valid < 0).mean() * 100
                else:
                    return (valid > 0).mean() * 100

            def calc_avg_return(returns_col):
                valid = type_df[returns_col].dropna()
                if len(valid) == 0:
                    return 0.0
                return valid.mean()

            # Find best and worst
            valid_30d = type_df.dropna(subset=['return_30d'])
            if len(valid_30d) > 0:
                if is_sell:
                    best_idx = valid_30d['return_30d'].idxmin()
                    worst_idx = valid_30d['return_30d'].idxmax()
                else:
                    best_idx = valid_30d['return_30d'].idxmax()
                    worst_idx = valid_30d['return_30d'].idxmin()

                best_ticker = valid_30d.loc[best_idx, 'ticker']
                best_return = valid_30d.loc[best_idx, 'return_30d']
                worst_ticker = valid_30d.loc[worst_idx, 'ticker']
                worst_return = valid_30d.loc[worst_idx, 'return_30d']
            else:
                best_ticker = "N/A"
                best_return = 0
                worst_ticker = "N/A"
                worst_return = 0

            results[signal_type] = SignalPerformance(
                signal_type=signal_type,
                total_signals=len(type_df),
                win_rate_5d=calc_win_rate('return_5d'),
                win_rate_10d=calc_win_rate('return_10d'),
                win_rate_30d=calc_win_rate('return_30d'),
                avg_return_5d=calc_avg_return('return_5d'),
                avg_return_10d=calc_avg_return('return_10d'),
                avg_return_30d=calc_avg_return('return_30d'),
                best_ticker=best_ticker,
                best_return=best_return,
                worst_ticker=worst_ticker,
                worst_return=worst_return
            )

        return results

    def get_performance_summary(self, days_back: int = 90) -> Dict[str, any]:
        """Get overall performance summary."""
        df = self.get_signals_with_returns(days_back)

        if df.empty:
            return {
                'total_signals': 0,
                'signals_with_returns': 0,
                'overall_win_rate': 0,
                'overall_avg_return': 0,
                'by_signal_type': {}
            }

        # Overall metrics (for BUY signals)
        buy_signals = df[df['signal_type'].str.contains('BUY', case=False, na=False)]
        valid_returns = buy_signals['return_10d'].dropna()

        return {
            'total_signals': len(df),
            'signals_with_returns': len(df.dropna(subset=['return_10d'])),
            'overall_win_rate': (valid_returns > 0).mean() * 100 if len(valid_returns) > 0 else 0,
            'overall_avg_return': valid_returns.mean() if len(valid_returns) > 0 else 0,
            'by_signal_type': self.get_performance_by_signal_type(days_back),
            'date_range': {
                'start': df['signal_date'].min().strftime('%Y-%m-%d') if len(df) > 0 else None,
                'end': df['signal_date'].max().strftime('%Y-%m-%d') if len(df) > 0 else None
            }
        }

    def get_recent_signals_performance(self, limit: int = 50) -> pd.DataFrame:
        """Get recent signals with their actual performance."""
        df = self.get_signals_with_returns(days_back=60)

        if df.empty:
            return df

        # Add win/loss indicator for 5d
        df['result_5d'] = df.apply(
            lambda x: '✅ Win' if (x['return_5d'] and x['return_5d'] > 0 and 'BUY' in str(x['signal_type']).upper()) or
                                 (x['return_5d'] and x['return_5d'] < 0 and 'SELL' in str(x['signal_type']).upper())
                      else '❌ Loss' if x['return_5d'] is not None else '⏳ Pending',
            axis=1
        )

        # Sort by date and limit
        df = df.sort_values('signal_date', ascending=False).head(limit)

        # Return columns (include return_2d if available)
        cols = ['ticker', 'signal_date', 'signal_type', 'signal_strength']
        if 'return_2d' in df.columns:
            cols.append('return_2d')
        cols.extend(['return_5d', 'return_10d', 'return_30d', 'result_5d'])

        return df[[c for c in cols if c in df.columns]]

    def save_signal_returns(self, signals_df: pd.DataFrame):
        """Save calculated returns back to database."""
        if signals_df.empty:
            return

        with get_connection() as conn:
            with conn.cursor() as cur:
                for _, row in signals_df.iterrows():
                    if pd.notna(row.get('return_5d')) or pd.notna(row.get('return_10d')):
                        cur.execute("""
                            UPDATE trading_signals
                            SET return_5d = %s, return_10d = %s, return_30d = %s
                            WHERE ticker = %s AND date = %s
                        """, (
                            row.get('return_5d'),
                            row.get('return_10d'),
                            row.get('return_30d'),
                            row['ticker'],
                            row['signal_date']
                        ))
                conn.commit()

        logger.info(f"Saved returns for {len(signals_df)} signals")


# =============================================================================
# INSTITUTIONAL SIGNAL PERFORMANCE (Phase 2-4)
# =============================================================================

@dataclass
class InstitutionalSignalPerformance:
    """Performance metrics for an institutional signal type."""
    signal_name: str
    signal_value: str  # e.g., "BULLISH", "CEO_BOUGHT", "BUFFETT_ADDED"
    total_occurrences: int
    win_rate_5d: float
    win_rate_10d: float
    avg_return_5d: float
    avg_return_10d: float
    best_ticker: str = ""
    best_return: float = 0.0
    worst_ticker: str = ""
    worst_return: float = 0.0


class InstitutionalSignalTracker:
    """
    Track performance of institutional signals (GEX, Insider, 13F, etc.)

    Reads from signal_snapshots table where full_signal_json contains
    all institutional signal values.
    """

    def __init__(self):
        self.engine = get_engine()

    def get_institutional_signals_with_returns(self, days_back: int = 90) -> pd.DataFrame:
        """
        Get institutional signals from signal_snapshots with calculated returns.

        Returns DataFrame with all institutional signal fields + forward returns.
        """
        import json

        query = """
                SELECT ticker, \
                       snapshot_date, \
                       price_at_snapshot, \
                       full_signal_json::text as full_signal_json, today_signal, \
                       today_score
                FROM signal_snapshots
                WHERE snapshot_date >= CURRENT_DATE - INTERVAL '%s days'
                  AND snapshot_date <= CURRENT_DATE - INTERVAL '5 days'
                  AND full_signal_json IS NOT NULL
                  AND length (full_signal_json::text) \
                    > 2
                ORDER BY snapshot_date DESC
                """

        try:
            df = pd.read_sql(query % days_back, self.engine)

            if df.empty:
                logger.warning("No signal snapshots found")
                return pd.DataFrame()

            logger.info(f"Found {len(df)} signal snapshots")

            # Parse JSON and extract institutional signals
            institutional_data = []

            for _, row in df.iterrows():
                try:
                    signal_data = json.loads(row['full_signal_json'])

                    inst_row = {
                        'ticker': row['ticker'],
                        'snapshot_date': row['snapshot_date'],
                        'price_at_snapshot': row['price_at_snapshot'],
                        'today_signal': row['today_signal'],
                        'today_score': row['today_score'],
                        # GEX
                        'gex_signal': signal_data.get('gex_signal', 'NEUTRAL'),
                        'gex_score': signal_data.get('gex_score', 50),
                        'gex_regime': signal_data.get('gex_regime', 'NEUTRAL'),
                        # Dark Pool
                        'dark_pool_signal': signal_data.get('dark_pool_signal', 'NEUTRAL'),
                        'dark_pool_score': signal_data.get('dark_pool_score', 50),
                        'institutional_bias': signal_data.get('institutional_bias', 'NEUTRAL'),
                        # Cross-Asset
                        'cross_asset_signal': signal_data.get('cross_asset_signal', 'NEUTRAL'),
                        'cycle_phase': signal_data.get('cycle_phase', ''),
                        # Insider
                        'insider_signal': signal_data.get('insider_signal', 'NEUTRAL'),
                        'insider_score': signal_data.get('insider_score', 50),
                        'insider_ceo_bought': signal_data.get('insider_ceo_bought', False),
                        'insider_cfo_bought': signal_data.get('insider_cfo_bought', False),
                        'insider_cluster_buying': signal_data.get('insider_cluster_buying', False),
                        'insider_cluster_selling': signal_data.get('insider_cluster_selling', False),
                        # 13F
                        'inst_13f_signal': signal_data.get('inst_13f_signal', 'NEUTRAL'),
                        'inst_13f_score': signal_data.get('inst_13f_score', 50),
                        'inst_buffett_owns': signal_data.get('inst_buffett_owns', False),
                        'inst_buffett_added': signal_data.get('inst_buffett_added', False),
                        'inst_activist_involved': signal_data.get('inst_activist_involved', False),
                        # Whisper
                        'whisper_signal': signal_data.get('whisper_signal', 'NEUTRAL'),
                        'whisper_score': signal_data.get('whisper_score', 50),
                    }
                    institutional_data.append(inst_row)

                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug(f"Could not parse JSON for {row['ticker']}: {e}")
                    continue

            if not institutional_data:
                return pd.DataFrame()

            inst_df = pd.DataFrame(institutional_data)

            # Calculate forward returns
            inst_df = self._calculate_forward_returns(inst_df)

            return inst_df

        except Exception as e:
            logger.error(f"Error getting institutional signals: {e}")
            return pd.DataFrame()

    def _calculate_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate 5d and 10d forward returns for each signal."""
        if df.empty:
            return df

        # Get unique tickers
        tickers = df['ticker'].unique().tolist()

        # Date range
        min_date = pd.to_datetime(df['snapshot_date'].min()) - timedelta(days=5)
        max_date = pd.to_datetime(df['snapshot_date'].max()) + timedelta(days=15)

        # Fetch prices
        try:
            prices = yf.download(
                tickers,
                start=min_date.strftime('%Y-%m-%d'),
                end=max_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True
            )

            if prices.empty:
                df['return_5d'] = None
                df['return_10d'] = None
                return df

            if len(tickers) == 1:
                price_df = prices['Close'].to_frame(tickers[0])
            else:
                price_df = prices['Close']

        except Exception as e:
            logger.warning(f"Could not fetch prices: {e}")
            df['return_5d'] = None
            df['return_10d'] = None
            return df

        # Calculate returns
        returns_5d = []
        returns_10d = []

        for _, row in df.iterrows():
            ticker = row['ticker']
            signal_date = pd.to_datetime(row['snapshot_date']).date()

            try:
                if ticker not in price_df.columns:
                    returns_5d.append(None)
                    returns_10d.append(None)
                    continue

                ticker_prices = price_df[ticker].dropna()

                # Find price on signal date
                signal_price = row['price_at_snapshot']
                if signal_price <= 0:
                    mask = ticker_prices.index.date >= signal_date
                    if mask.any():
                        signal_price = ticker_prices[mask].iloc[0]

                # 5-day return
                target_5d = signal_date + timedelta(days=5)
                mask_5d = ticker_prices.index.date >= target_5d
                if mask_5d.any() and signal_price > 0:
                    price_5d = ticker_prices[mask_5d].iloc[0]
                    returns_5d.append((price_5d - signal_price) / signal_price * 100)
                else:
                    returns_5d.append(None)

                # 10-day return
                target_10d = signal_date + timedelta(days=10)
                mask_10d = ticker_prices.index.date >= target_10d
                if mask_10d.any() and signal_price > 0:
                    price_10d = ticker_prices[mask_10d].iloc[0]
                    returns_10d.append((price_10d - signal_price) / signal_price * 100)
                else:
                    returns_10d.append(None)

            except Exception as e:
                logger.debug(f"Return calc error for {ticker}: {e}")
                returns_5d.append(None)
                returns_10d.append(None)

        df['return_5d'] = returns_5d
        df['return_10d'] = returns_10d

        return df

    def get_performance_by_institutional_signal(self, days_back: int = 90) -> Dict[
        str, List[InstitutionalSignalPerformance]]:
        """
        Get performance breakdown for each institutional signal type.

        Returns dict with signal category -> list of performance by signal value.
        """
        df = self.get_institutional_signals_with_returns(days_back)

        if df.empty:
            return {}

        results = {}

        # Define signals to track
        signal_configs = [
            ('GEX Signal', 'gex_signal', ['BULLISH', 'BEARISH', 'NEUTRAL', 'PINNED']),
            ('GEX Regime', 'gex_regime', ['POSITIVE_GEX', 'NEGATIVE_GEX', 'NEUTRAL']),
            ('Dark Pool', 'dark_pool_signal', ['ACCUMULATION', 'DISTRIBUTION', 'NEUTRAL']),
            ('Institutional Bias', 'institutional_bias', ['BUYING', 'SELLING', 'NEUTRAL']),
            ('Cross-Asset', 'cross_asset_signal', ['RISK_ON', 'RISK_OFF', 'NEUTRAL']),
            ('Insider Signal', 'insider_signal', ['STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL']),
            ('13F Signal', 'inst_13f_signal', ['STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL']),
            ('Whisper Signal', 'whisper_signal', ['BEAT_EXPECTED', 'MISS_EXPECTED', 'NEUTRAL']),
        ]

        # Boolean signals
        bool_signals = [
            ('Insider CEO Bought', 'insider_ceo_bought'),
            ('Insider CFO Bought', 'insider_cfo_bought'),
            ('Insider Cluster Buying', 'insider_cluster_buying'),
            ('Insider Cluster Selling', 'insider_cluster_selling'),
            ('Buffett Owns', 'inst_buffett_owns'),
            ('Buffett Added', 'inst_buffett_added'),
            ('Activist Involved', 'inst_activist_involved'),
        ]

        # Process categorical signals
        for signal_name, column, values in signal_configs:
            performances = []

            for value in values:
                subset = df[df[column] == value]

                if len(subset) < 3:  # Need at least 3 occurrences
                    continue

                valid_5d = subset['return_5d'].dropna()
                valid_10d = subset['return_10d'].dropna()

                perf = InstitutionalSignalPerformance(
                    signal_name=signal_name,
                    signal_value=value,
                    total_occurrences=len(subset),
                    win_rate_5d=(valid_5d > 0).mean() * 100 if len(valid_5d) > 0 else 0,
                    win_rate_10d=(valid_10d > 0).mean() * 100 if len(valid_10d) > 0 else 0,
                    avg_return_5d=valid_5d.mean() if len(valid_5d) > 0 else 0,
                    avg_return_10d=valid_10d.mean() if len(valid_10d) > 0 else 0,
                )

                # Best/worst
                if len(valid_10d) > 0:
                    best_idx = subset['return_10d'].idxmax()
                    worst_idx = subset['return_10d'].idxmin()
                    perf.best_ticker = subset.loc[best_idx, 'ticker']
                    perf.best_return = subset.loc[best_idx, 'return_10d']
                    perf.worst_ticker = subset.loc[worst_idx, 'ticker']
                    perf.worst_return = subset.loc[worst_idx, 'return_10d']

                performances.append(perf)

            if performances:
                results[signal_name] = performances

        # Process boolean signals
        for signal_name, column in bool_signals:
            if column not in df.columns:
                continue

            performances = []

            for value in [True, False]:
                subset = df[df[column] == value]

                if len(subset) < 3:
                    continue

                valid_5d = subset['return_5d'].dropna()
                valid_10d = subset['return_10d'].dropna()

                value_str = "YES" if value else "NO"

                perf = InstitutionalSignalPerformance(
                    signal_name=signal_name,
                    signal_value=value_str,
                    total_occurrences=len(subset),
                    win_rate_5d=(valid_5d > 0).mean() * 100 if len(valid_5d) > 0 else 0,
                    win_rate_10d=(valid_10d > 0).mean() * 100 if len(valid_10d) > 0 else 0,
                    avg_return_5d=valid_5d.mean() if len(valid_5d) > 0 else 0,
                    avg_return_10d=valid_10d.mean() if len(valid_10d) > 0 else 0,
                )

                if len(valid_10d) > 0:
                    best_idx = subset['return_10d'].idxmax()
                    worst_idx = subset['return_10d'].idxmin()
                    perf.best_ticker = subset.loc[best_idx, 'ticker']
                    perf.best_return = subset.loc[best_idx, 'return_10d']
                    perf.worst_ticker = subset.loc[worst_idx, 'ticker']
                    perf.worst_return = subset.loc[worst_idx, 'return_10d']

                performances.append(perf)

            if performances:
                results[signal_name] = performances

        return results

    def get_best_institutional_signals(self, days_back: int = 90, top_n: int = 10) -> pd.DataFrame:
        """
        Get the best performing institutional signals ranked by win rate.

        Returns DataFrame with signal name, value, win rate, avg return.
        """
        performance = self.get_performance_by_institutional_signal(days_back)

        if not performance:
            return pd.DataFrame()

        rows = []
        for signal_name, perfs in performance.items():
            for p in perfs:
                if p.total_occurrences >= 5:  # Minimum sample size
                    rows.append({
                        'Signal': signal_name,
                        'Value': p.signal_value,
                        'Count': p.total_occurrences,
                        'Win Rate 5d': f"{p.win_rate_5d:.1f}%",
                        'Win Rate 10d': f"{p.win_rate_10d:.1f}%",
                        'Avg Return 5d': f"{p.avg_return_5d:+.2f}%",
                        'Avg Return 10d': f"{p.avg_return_10d:+.2f}%",
                        '_sort_key': p.win_rate_10d,
                    })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values('_sort_key', ascending=False).head(top_n)
        df = df.drop('_sort_key', axis=1)

        return df

    def get_institutional_signal_summary(self, days_back: int = 90) -> Dict[str, any]:
        """Get summary of institutional signal performance."""
        df = self.get_institutional_signals_with_returns(days_back)

        if df.empty:
            return {
                'total_snapshots': 0,
                'date_range': None,
                'best_signals': [],
            }

        best_df = self.get_best_institutional_signals(days_back, top_n=5)

        return {
            'total_snapshots': len(df),
            'snapshots_with_returns': len(df.dropna(subset=['return_10d'])),
            'date_range': {
                'start': df['snapshot_date'].min().strftime('%Y-%m-%d'),
                'end': df['snapshot_date'].max().strftime('%Y-%m-%d'),
            },
            'best_signals': best_df.to_dict('records') if not best_df.empty else [],
            'performance_by_signal': self.get_performance_by_institutional_signal(days_back),
        }

    def print_leaderboard(self, days_back: int = 90):
        """Print institutional signal leaderboard to console."""
        print("\n" + "=" * 70)
        print("🏆 INSTITUTIONAL SIGNAL LEADERBOARD")
        print("=" * 70)

        best = self.get_best_institutional_signals(days_back, top_n=15)

        if best.empty:
            print("No data available yet. Run signal generation to populate.")
            return

        print(f"\nTop signals by 10-day win rate (last {days_back} days):\n")
        print(best.to_string(index=False))

        # Highlight key findings
        print("\n" + "-" * 70)
        print("📊 KEY FINDINGS:")

        performance = self.get_performance_by_institutional_signal(days_back)

        # CEO bought
        if 'Insider CEO Bought' in performance:
            ceo_perfs = performance['Insider CEO Bought']
            for p in ceo_perfs:
                if p.signal_value == 'YES' and p.total_occurrences >= 3:
                    print(
                        f"   • CEO Bought: {p.win_rate_10d:.1f}% win rate, {p.avg_return_10d:+.2f}% avg ({p.total_occurrences} signals)")

        # Buffett added
        if 'Buffett Added' in performance:
            buff_perfs = performance['Buffett Added']
            for p in buff_perfs:
                if p.signal_value == 'YES' and p.total_occurrences >= 3:
                    print(
                        f"   • Buffett Added: {p.win_rate_10d:.1f}% win rate, {p.avg_return_10d:+.2f}% avg ({p.total_occurrences} signals)")

        # GEX Bullish
        if 'GEX Signal' in performance:
            gex_perfs = performance['GEX Signal']
            for p in gex_perfs:
                if p.signal_value == 'BULLISH' and p.total_occurrences >= 5:
                    print(
                        f"   • GEX Bullish: {p.win_rate_10d:.1f}% win rate, {p.avg_return_10d:+.2f}% avg ({p.total_occurrences} signals)")

        print("=" * 70)


# Convenience functions
def get_signal_performance_tracker() -> SignalPerformanceTracker:
    """Get signal performance tracker instance."""
    return SignalPerformanceTracker()


def get_institutional_signal_tracker() -> InstitutionalSignalTracker:
    """Get institutional signal tracker instance."""
    return InstitutionalSignalTracker()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--institutional':
        # Test institutional signals
        print("\n🏛️ Testing Institutional Signal Tracker...")
        inst_tracker = InstitutionalSignalTracker()
        inst_tracker.print_leaderboard(days_back=90)
    else:
        # Test regular signals
        tracker = SignalPerformanceTracker()

        print("\n=== Signal Performance Summary ===")
        summary = tracker.get_performance_summary(days_back=60)
        print(f"Total signals: {summary['total_signals']}")
        print(f"With returns: {summary['signals_with_returns']}")
        print(f"Overall win rate: {summary['overall_win_rate']:.1f}%")
        print(f"Overall avg return: {summary['overall_avg_return']:.2f}%")

        print("\n=== By Signal Type ===")
        for signal_type, perf in summary['by_signal_type'].items():
            print(f"\n{signal_type}:")
            print(f"  Signals: {perf.total_signals}")
            print(f"  Win Rate (10d): {perf.win_rate_10d:.1f}%")
            print(f"  Avg Return (10d): {perf.avg_return_10d:.2f}%")
            print(f"  Best: {perf.best_ticker} ({perf.best_return:+.1f}%)")
            print(f"  Worst: {perf.worst_ticker} ({perf.worst_return:+.1f}%)")

        print("\n💡 Run with --institutional to see institutional signal performance")