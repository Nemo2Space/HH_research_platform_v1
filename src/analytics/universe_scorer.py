"""
Universe Scoring Integration

Calculates Options Flow and Short Squeeze scores for stocks in the screener universe.
These scores are normalized to 0-100 and saved to screener_scores table.

Score Interpretation:
- options_flow_score: 0-100 (50=neutral, >70=bullish, <30=bearish)
- short_squeeze_score: 0-100 (>70=high squeeze potential, >50=moderate, <30=low)

Author: Alpha Research Platform
"""

import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Import analyzers
try:
    from src.analytics.options_flow import OptionsFlowAnalyzer
    OPTIONS_FLOW_AVAILABLE = True
except ImportError:
    OPTIONS_FLOW_AVAILABLE = False
    logger.warning("OptionsFlowAnalyzer not available")

try:
    from src.analytics.short_squeeze import ShortSqueezeDetector
    SHORT_SQUEEZE_AVAILABLE = True
except ImportError:
    SHORT_SQUEEZE_AVAILABLE = False
    logger.warning("ShortSqueezeDetector not available")

try:
    from src.db.connection import get_connection, get_engine
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


@dataclass
class UniverseScores:
    """Scores for a single ticker."""
    ticker: str
    options_flow_score: Optional[float] = None  # 0-100 (50=neutral)
    options_sentiment: Optional[str] = None  # BULLISH, BEARISH, NEUTRAL
    short_squeeze_score: Optional[float] = None  # 0-100
    squeeze_risk: Optional[str] = None  # HIGH, MODERATE, LOW, MINIMAL
    error: Optional[str] = None


class UniverseScorer:
    """
    Calculates Options Flow and Short Squeeze scores for universe stocks.
    """

    def __init__(self, skip_ibkr: bool = False):
        """Initialize analyzers."""
        self.options_analyzer = OptionsFlowAnalyzer() if OPTIONS_FLOW_AVAILABLE else None
        self.squeeze_detector = ShortSqueezeDetector() if SHORT_SQUEEZE_AVAILABLE else None
        self.skip_ibkr = skip_ibkr

    def _normalize_options_score(self, sentiment_score: float) -> float:
        """
        Normalize options sentiment score from -100 to +100 range to 0-100 range.

        Input: -100 (very bearish) to +100 (very bullish)
        Output: 0 (very bearish) to 100 (very bullish), 50 = neutral
        """
        # sentiment_score is typically -100 to +100
        # Convert to 0-100 where 50 is neutral
        normalized = (sentiment_score + 100) / 2
        return max(0, min(100, normalized))

    def score_ticker(self, ticker: str,
                     include_options: bool = True,
                     include_squeeze: bool = True) -> UniverseScores:
        """
        Calculate all scores for a single ticker.

        Args:
            ticker: Stock symbol
            include_options: Whether to calculate options flow score
            include_squeeze: Whether to calculate short squeeze score

        Returns:
            UniverseScores with all available scores
        """
        scores = UniverseScores(ticker=ticker)

        try:
            # Options Flow Score
            if include_options and self.options_analyzer:
                try:
                    options_summary = self.options_analyzer.analyze_ticker(ticker, skip_ibkr=self.skip_ibkr)
                    if options_summary and options_summary.total_call_volume > 0:
                        # Normalize from -100/+100 to 0-100
                        scores.options_flow_score = self._normalize_options_score(
                            options_summary.sentiment_score
                        )
                        scores.options_sentiment = options_summary.overall_sentiment
                except Exception as e:
                    logger.debug(f"Options flow error for {ticker}: {e}")

            # Short Squeeze Score
            if include_squeeze and self.squeeze_detector:
                try:
                    squeeze_data = self.squeeze_detector.analyze_ticker(ticker)
                    if squeeze_data and squeeze_data.short_percent_of_float is not None:
                        scores.short_squeeze_score = squeeze_data.squeeze_score
                        scores.squeeze_risk = squeeze_data.squeeze_risk
                except Exception as e:
                    logger.debug(f"Short squeeze error for {ticker}: {e}")

        except Exception as e:
            scores.error = str(e)
            logger.warning(f"Error scoring {ticker}: {e}")

        return scores

    def score_universe(self, tickers: List[str],
                       max_workers: int = 3,
                       include_options: bool = True,
                       include_squeeze: bool = True) -> List[UniverseScores]:
        """
        Calculate scores for multiple tickers in parallel.

        Args:
            tickers: List of stock symbols
            max_workers: Number of parallel threads
            include_options: Whether to calculate options flow
            include_squeeze: Whether to calculate short squeeze

        Returns:
            List of UniverseScores
        """
        results = []

        logger.info(f"Scoring {len(tickers)} tickers (options={include_options}, squeeze={include_squeeze})")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.score_ticker, ticker, include_options, include_squeeze
                ): ticker
                for ticker in tickers
            }

            for i, future in enumerate(as_completed(futures)):
                ticker = futures[future]
                try:
                    score = future.result()
                    results.append(score)

                    if (i + 1) % 10 == 0:
                        logger.info(f"Scored {i + 1}/{len(tickers)} tickers")

                except Exception as e:
                    logger.error(f"Error scoring {ticker}: {e}")
                    results.append(UniverseScores(ticker=ticker, error=str(e)))

        logger.info(f"Completed scoring {len(results)} tickers")
        return results

    def update_screener_scores(self, scores: List[UniverseScores],
                               target_date: date = None) -> int:
        """
        Update screener_scores table with options flow and squeeze scores.

        Args:
            scores: List of UniverseScores
            target_date: Date to update (default: latest date for each ticker)

        Returns:
            Number of rows updated
        """
        if not DB_AVAILABLE:
            logger.warning("Database not available, cannot update scores")
            return 0

        def to_native(val):
            """Convert numpy types to native Python types."""
            if val is None:
                return None
            try:
                import numpy as np
                if isinstance(val, (np.integer, np.floating)):
                    return float(val)
                if isinstance(val, np.ndarray):
                    return val.tolist()
            except ImportError:
                pass
            return val

        updated = 0

        with get_connection() as conn:
            with conn.cursor() as cur:
                for score in scores:
                    try:
                        # Only update if we have at least one score
                        if score.options_flow_score is None and score.short_squeeze_score is None:
                            continue

                        # Build dynamic update query based on what scores we have
                        updates = []
                        values = []

                        if score.options_flow_score is not None:
                            updates.append("options_flow_score = %s")
                            values.append(to_native(score.options_flow_score))
                            updates.append("options_sentiment = %s")
                            values.append(score.options_sentiment)

                        if score.short_squeeze_score is not None:
                            updates.append("short_squeeze_score = %s")
                            values.append(to_native(score.short_squeeze_score))
                            updates.append("squeeze_risk = %s")
                            values.append(score.squeeze_risk)

                        if not updates:
                            continue

                        # Add ticker for WHERE clause
                        values.append(score.ticker)

                        # Update specific date if provided, otherwise latest
                        if target_date:
                            query = f"""
                                UPDATE screener_scores 
                                SET {', '.join(updates)}
                                WHERE ticker = %s 
                                  AND date = %s
                            """
                            values.append(target_date)
                        else:
                            # Update the LATEST record for this ticker
                            query = f"""
                                UPDATE screener_scores 
                                SET {', '.join(updates)}
                                WHERE ticker = %s 
                                  AND date = (
                                      SELECT MAX(date) FROM screener_scores WHERE ticker = %s
                                  )
                            """
                            # Need ticker twice for subquery
                            values.append(score.ticker)

                        cur.execute(query, values)

                        if cur.rowcount > 0:
                            updated += 1

                    except Exception as e:
                        logger.warning(f"Error updating {score.ticker}: {e}")

                conn.commit()

        logger.info(f"Updated {updated} screener_scores with flow/squeeze data")
        return updated

    def score_and_save_universe(self, tickers: List[str] = None,
                                max_workers: int = 3) -> Tuple[List[UniverseScores], int]:
        """
        Score tickers and save to database in one step.

        Args:
            tickers: List of tickers (if None, loads from screener_scores)
            max_workers: Parallel workers

        Returns:
            Tuple of (scores list, rows updated)
        """
        # Load tickers from screener if not provided
        if tickers is None:
            tickers = self._load_universe_tickers()

        if not tickers:
            logger.warning("No tickers to score")
            return [], 0

        # Score all tickers
        scores = self.score_universe(tickers, max_workers=max_workers)

        # Save to database (updates latest record for each ticker)
        updated = self.update_screener_scores(scores)

        return scores, updated

    def _load_universe_tickers(self) -> List[str]:
        """Load tickers from screener_scores (most recent 7 days)."""
        if not DB_AVAILABLE:
            return []

        try:
            query = """
                SELECT DISTINCT ticker FROM screener_scores 
                WHERE date >= CURRENT_DATE - INTERVAL '7 days'
                ORDER BY ticker
            """
            df = pd.read_sql(query, get_engine())
            return df['ticker'].tolist()
        except Exception as e:
            logger.error(f"Error loading universe: {e}")
            return []


# Convenience functions
def score_universe(tickers: List[str], max_workers: int = 3) -> List[UniverseScores]:
    """Score a list of tickers for options flow and short squeeze potential."""
    scorer = UniverseScorer(skip_ibkr=True)
    return scorer.score_universe(tickers, max_workers=max_workers)


def update_universe_scores(max_workers: int = 3) -> int:
    """Update all tickers in screener_scores with flow/squeeze scores."""
    scorer = UniverseScorer(skip_ibkr=True)
    _, updated = scorer.score_and_save_universe(max_workers=max_workers)
    return updated


def get_top_options_flow(limit: int = 20, min_score: float = 60) -> pd.DataFrame:
    """Get top stocks by options flow score (bullish signals)."""
    if not DB_AVAILABLE:
        return pd.DataFrame()

    query = """
        SELECT DISTINCT ON (ticker) ticker, date, options_flow_score, options_sentiment,
               sentiment_score, composite_score, total_score
        FROM screener_scores
        WHERE options_flow_score >= %s
        ORDER BY ticker, date DESC
    """
    df = pd.read_sql(query, get_engine(), params=[min_score])
    return df.sort_values('options_flow_score', ascending=False).head(limit)


def get_top_squeeze_candidates(limit: int = 20, min_score: float = 50) -> pd.DataFrame:
    """Get top stocks by short squeeze score."""
    if not DB_AVAILABLE:
        return pd.DataFrame()

    query = """
        SELECT DISTINCT ON (ticker) ticker, date, short_squeeze_score, squeeze_risk,
               sentiment_score, composite_score, total_score
        FROM screener_scores
        WHERE short_squeeze_score >= %s
        ORDER BY ticker, date DESC
    """
    df = pd.read_sql(query, get_engine(), params=[min_score])
    return df.sort_values('short_squeeze_score', ascending=False).head(limit)