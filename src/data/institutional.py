"""
Alpha Platform - Institutional Holdings Data

Fetches institutional holdings from SEC 13F filings via Finnhub.
"""

import os
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

from src.db.connection import get_connection
from src.db.repository import Repository
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InstitutionalHolding:
    """Institutional holding data."""
    ticker: str
    institution_cik: str
    institution_name: str
    report_date: str
    filing_date: str
    shares_held: int
    market_value: int
    shares_change: int
    shares_change_pct: float
    position_type: str  # NEW, INCREASED, DECREASED, UNCHANGED, SOLD_OUT


class InstitutionalDataFetcher:
    """Fetches institutional holdings data."""

    def __init__(self, repository: Optional[Repository] = None):
        self.repo = repository or Repository()
        self.finnhub_key = os.getenv("FINNHUB_API_KEY", "")

    def get_finnhub_institutional(self, ticker: str) -> List[InstitutionalHolding]:
        """Fetch institutional ownership from Finnhub."""
        if not self.finnhub_key:
            logger.debug(f"{ticker}: Finnhub API key not configured")
            return []

        try:
            url = "https://finnhub.io/api/v1/institutional-ownership"
            params = {
                "symbol": ticker,
                "token": self.finnhub_key
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            ownership_data = data.get('data', [])

            if not ownership_data:
                logger.debug(f"{ticker}: No institutional data from Finnhub")
                return []

            results = []
            # Get latest quarter data
            latest = ownership_data[0] if ownership_data else {}
            holders = latest.get('ownership', [])
            filing_date = latest.get('filingDate', '')
            report_date = filing_date[:10] if filing_date else ''

            for holder in holders[:50]:  # Top 50 holders
                shares = holder.get('share', 0) or 0
                change = holder.get('change', 0) or 0
                value = holder.get('value', 0) or 0

                # Calculate change percentage
                prev_shares = shares - change
                if prev_shares > 0:
                    change_pct = (change / prev_shares) * 100
                elif change > 0:
                    change_pct = 100.0
                else:
                    change_pct = 0.0

                # Determine position type
                if prev_shares == 0 and shares > 0:
                    position_type = 'NEW'
                elif shares == 0 and prev_shares > 0:
                    position_type = 'SOLD_OUT'
                elif change > 0:
                    position_type = 'INCREASED'
                elif change < 0:
                    position_type = 'DECREASED'
                else:
                    position_type = 'UNCHANGED'

                # Use name as CIK if not available
                cik = holder.get('cik', '') or holder.get('name', 'Unknown')[:20]

                results.append(InstitutionalHolding(
                    ticker=ticker,
                    institution_cik=str(cik)[:20],
                    institution_name=holder.get('name', 'Unknown')[:300],
                    report_date=report_date,
                    filing_date=filing_date[:10] if filing_date else None,
                    shares_held=shares,
                    market_value=value,
                    shares_change=change,
                    shares_change_pct=round(change_pct, 4),
                    position_type=position_type
                ))

            logger.info(f"{ticker}: Found {len(results)} institutional holders")
            return results

        except Exception as e:
            logger.error(f"{ticker}: Finnhub institutional error - {e}")
            return []

    def save_holding(self, holding: InstitutionalHolding) -> bool:
        """Save institutional holding to database."""
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO institutional_holdings (
                            ticker, institution_cik, institution_name, report_date,
                            filing_date, shares_held, market_value, shares_change,
                            shares_change_pct, position_type
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, institution_cik, report_date) 
                        DO UPDATE SET
                            shares_held = EXCLUDED.shares_held,
                            market_value = EXCLUDED.market_value,
                            shares_change = EXCLUDED.shares_change,
                            shares_change_pct = EXCLUDED.shares_change_pct,
                            position_type = EXCLUDED.position_type
                    """, (
                        holding.ticker,
                        holding.institution_cik,
                        holding.institution_name,
                        holding.report_date,
                        holding.filing_date,
                        holding.shares_held,
                        holding.market_value,
                        holding.shares_change,
                        holding.shares_change_pct,
                        holding.position_type
                    ))
            return True
        except Exception as e:
            logger.error(f"Error saving holding: {e}")
            return False

    def fetch_and_save(self, ticker: str) -> int:
        """Fetch and save institutional holdings for a ticker."""
        holdings = self.get_finnhub_institutional(ticker)

        saved = 0
        for holding in holdings:
            if self.save_holding(holding):
                saved += 1

        return saved

    def get_institutional_signal(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze institutional holdings and generate a signal.

        Returns:
            Dict with institutional_signal score (0-100) and details
        """
        query = """
            SELECT institution_name, shares_held, market_value, 
                   shares_change, shares_change_pct, position_type
            FROM institutional_holdings
            WHERE ticker = %(ticker)s
            ORDER BY market_value DESC NULLS LAST
            LIMIT 30
        """

        try:
            df = pd.read_sql(query, self.repo.engine, params={"ticker": ticker})
        except:
            df = pd.DataFrame()

        if len(df) == 0:
            return {
                'institutional_signal': 50,
                'total_institutions': 0,
                'buyers': 0,
                'sellers': 0,
                'new_positions': 0,
                'sold_out': 0,
                'net_change': 0,
                'summary': 'No institutional data'
            }

        # Count by position type
        buyers = len(df[df['position_type'].isin(['NEW', 'INCREASED'])])
        sellers = len(df[df['position_type'].isin(['DECREASED', 'SOLD_OUT'])])
        new_positions = len(df[df['position_type'] == 'NEW'])
        sold_out = len(df[df['position_type'] == 'SOLD_OUT'])
        unchanged = len(df[df['position_type'] == 'UNCHANGED'])

        total = len(df)
        net_change = int(df['shares_change'].fillna(0).sum())

        # Calculate signal (0-100)
        if total == 0:
            signal = 50
        else:
            # Base score from buyer/seller ratio
            buyer_ratio = buyers / total
            signal = int(30 + buyer_ratio * 40)  # Range 30-70

            # Boost for new positions (bullish)
            if new_positions >= 3:
                signal = min(100, signal + 15)
            elif new_positions >= 1:
                signal = min(100, signal + 8)

            # Penalty for sold out positions (bearish)
            if sold_out >= 3:
                signal = max(0, signal - 15)
            elif sold_out >= 1:
                signal = max(0, signal - 8)

            # Adjust by net change
            avg_change_pct = df['shares_change_pct'].fillna(0).mean()
            if avg_change_pct > 10:
                signal = min(100, signal + 10)
            elif avg_change_pct < -10:
                signal = max(0, signal - 10)

        # Generate summary
        if buyers > sellers:
            summary = f"Accumulation: {buyers} buying, {sellers} selling"
        elif sellers > buyers:
            summary = f"Distribution: {sellers} selling, {buyers} buying"
        else:
            summary = f"Neutral: {buyers} buying, {sellers} selling"

        if new_positions > 0:
            summary += f", {new_positions} new"
        if sold_out > 0:
            summary += f", {sold_out} exited"

        return {
            'institutional_signal': signal,
            'total_institutions': total,
            'buyers': buyers,
            'sellers': sellers,
            'new_positions': new_positions,
            'sold_out': sold_out,
            'unchanged': unchanged,
            'net_change': net_change,
            'summary': summary
        }