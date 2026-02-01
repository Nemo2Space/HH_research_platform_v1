"""
Alpha Platform - Finviz Data

Fetches fundamental data, analyst ratings, and institutional ownership from Finviz.
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from src.db.connection import get_connection
from src.db.repository import Repository
from src.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from finvizfinance.quote import finvizfinance
    FINVIZ_AVAILABLE = True
except ImportError:
    FINVIZ_AVAILABLE = False
    logger.warning("finvizfinance not installed. Run: pip install finvizfinance")


class FinvizDataFetcher:
    """Fetches data from Finviz."""

    def __init__(self, repository: Optional[Repository] = None):
        self.repo = repository or Repository()

    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """
        Get fundamental data from Finviz.

        Returns dict with parsed fundamental metrics.
        """
        if not FINVIZ_AVAILABLE:
            return {}

        try:
            stock = finvizfinance(ticker)
            data = stock.ticker_fundament()

            if not data:
                logger.warning(f"{ticker}: No Finviz data")
                return {}

            # Parse key metrics
            result = {
                'ticker': ticker,
                # Ownership
                'inst_own': self._parse_pct(data.get('Inst Own')),
                'insider_own': self._parse_pct(data.get('Insider Own')),
                'short_float': self._parse_pct(data.get('Short Float')),
                # Valuation
                'pe': self._parse_float(data.get('P/E')),
                'forward_pe': self._parse_float(data.get('Forward P/E')),
                'peg': self._parse_float(data.get('PEG')),
                'ps': self._parse_float(data.get('P/S')),
                'pb': self._parse_float(data.get('P/B')),
                # Profitability
                'roe': self._parse_pct(data.get('ROE')),
                'roa': self._parse_pct(data.get('ROA')),
                'roi': self._parse_pct(data.get('ROI')),
                'profit_margin': self._parse_pct(data.get('Profit Margin')),
                'oper_margin': self._parse_pct(data.get('Oper. Margin')),
                'gross_margin': self._parse_pct(data.get('Gross Margin')),
                # Growth
                'eps': self._parse_float(data.get('EPS (ttm)')),
                'eps_growth_yoy': self._parse_pct(data.get('EPS Y/Y TTM')),
                'eps_growth_next_y': self._parse_pct(data.get('EPS next Y')),
                'sales_growth_yoy': self._parse_pct(data.get('Sales Y/Y TTM')),
                'revenue_growth': self._parse_pct(data.get('Sales Q/Q')),
                # Dividend
                'dividend_yield': self._parse_pct(data.get('Dividend %')),
                'payout_ratio': self._parse_pct(data.get('Payout')),
                # Price targets
                'target_price': self._parse_float(data.get('Target Price')),
                'price': self._parse_float(data.get('Price')),
                # Other
                'beta': self._parse_float(data.get('Beta')),
                'rsi': self._parse_float(data.get('RSI (14)')),
                'market_cap': data.get('Market Cap'),
                'employees': data.get('Employees'),
                # Debt
                'debt_eq': self._parse_float(data.get('Debt/Eq')),
                'current_ratio': self._parse_float(data.get('Current Ratio')),
                'quick_ratio': self._parse_float(data.get('Quick Ratio')),
                # Raw data for reference
                '_raw': data
            }

            logger.info(f"{ticker}: Finviz fundamentals fetched (Inst Own: {result['inst_own']}%)")
            return result

        except Exception as e:
            logger.error(f"{ticker}: Finviz error - {e}")
            return {}

    def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """Get analyst ratings from Finviz."""
        if not FINVIZ_AVAILABLE:
            return {}

        try:
            stock = finvizfinance(ticker)
            ratings_df = stock.ticker_outer_ratings()

            if ratings_df is None or ratings_df.empty:
                return {
                    'total_positive': 0,
                    'total_ratings': 0,
                    'buy_pct': 0
                }

            # Count positive ratings
            positive_ratings = ['Buy', 'Strong Buy', 'Overweight', 'Outperform', 'Positive']

            if 'Rating' in ratings_df.columns:
                buy_ratings = ratings_df[ratings_df['Rating'].isin(positive_ratings)]
                total_positive = len(buy_ratings)
                total_ratings = len(ratings_df)
                buy_pct = (total_positive / total_ratings * 100) if total_ratings > 0 else 0
            else:
                total_positive = 0
                total_ratings = 0
                buy_pct = 0

            logger.info(f"{ticker}: {total_positive}/{total_ratings} positive ratings")

            return {
                'total_positive': total_positive,
                'total_ratings': total_ratings,
                'buy_pct': round(buy_pct, 1)
            }

        except Exception as e:
            logger.error(f"{ticker}: Finviz ratings error - {e}")
            return {}

    def get_institutional_signal(self, ticker: str) -> Dict[str, Any]:
        """
        Calculate institutional signal from Finviz data.

        Uses institutional ownership % and changes to generate signal.
        """
        data = self.get_fundamentals(ticker)

        if not data:
            return {
                'institutional_signal': 50,
                'inst_own_pct': None,
                'insider_own_pct': None,
                'short_float_pct': None,
                'summary': 'No Finviz data'
            }

        inst_own = data.get('inst_own')
        insider_own = data.get('insider_own')
        short_float = data.get('short_float')

        # Calculate signal based on ownership metrics
        signal = 50  # Start neutral

        if inst_own is not None:
            # High institutional ownership is generally positive
            if inst_own >= 80:
                signal += 15
            elif inst_own >= 60:
                signal += 10
            elif inst_own >= 40:
                signal += 5
            elif inst_own < 20:
                signal -= 10

        if insider_own is not None:
            # Some insider ownership is good, very high might indicate control issues
            if 5 <= insider_own <= 30:
                signal += 10
            elif insider_own > 50:
                signal -= 5

        if short_float is not None:
            # High short interest is bearish
            if short_float >= 20:
                signal -= 15
            elif short_float >= 10:
                signal -= 10
            elif short_float >= 5:
                signal -= 5

        signal = max(0, min(100, signal))

        # Generate summary
        parts = []
        if inst_own is not None:
            parts.append(f"Inst: {inst_own:.1f}%")
        if insider_own is not None:
            parts.append(f"Insider: {insider_own:.1f}%")
        if short_float is not None:
            parts.append(f"Short: {short_float:.1f}%")

        summary = ", ".join(parts) if parts else "No ownership data"

        return {
            'institutional_signal': signal,
            'inst_own_pct': inst_own,
            'insider_own_pct': insider_own,
            'short_float_pct': short_float,
            'summary': summary
        }

    def _parse_float(self, value: Any) -> Optional[float]:
        """Parse a value to float, handling various formats."""
        if value is None or value == '-' or value == 'N/A' or value == '':
            return None
        try:
            if isinstance(value, (int, float)):
                return float(value)
            # Remove commas and parse
            clean = str(value).replace(',', '').replace('$', '')
            return float(clean)
        except (ValueError, TypeError):
            return None

    def _parse_pct(self, value: Any) -> Optional[float]:
        """Parse a percentage value."""
        if value is None or value == '-' or value == 'N/A' or value == '':
            return None
        try:
            if isinstance(value, (int, float)):
                return float(value)
            # Remove % and parse
            clean = str(value).replace('%', '').replace(',', '')
            return float(clean)
        except (ValueError, TypeError):
            return None

    def fetch_and_update_fundamentals(self, ticker: str) -> bool:
        """Fetch Finviz data and update fundamentals table."""
        data = self.get_fundamentals(ticker)

        if not data:
            return False

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Update fundamentals table with Finviz data
                    cur.execute("""
                        UPDATE fundamentals SET
                            pe_ratio = COALESCE(%s, pe_ratio),
                            forward_pe = COALESCE(%s, forward_pe),
                            peg_ratio = COALESCE(%s, peg_ratio),
                            price_to_sales = COALESCE(%s, price_to_sales),
                            price_to_book = COALESCE(%s, price_to_book),
                            roe = COALESCE(%s, roe),
                            roa = COALESCE(%s, roa),
                            profit_margin = COALESCE(%s, profit_margin),
                            operating_margin = COALESCE(%s, operating_margin),
                            gross_margin = COALESCE(%s, gross_margin),
                            dividend_yield = COALESCE(%s, dividend_yield),
                            beta = COALESCE(%s, beta),
                            updated_at = NOW()
                        WHERE ticker = %s
                    """, (
                        data.get('pe'),
                        data.get('forward_pe'),
                        data.get('peg'),
                        data.get('ps'),
                        data.get('pb'),
                        data.get('roe'),
                        data.get('roa'),
                        data.get('profit_margin'),
                        data.get('oper_margin'),
                        data.get('gross_margin'),
                        data.get('dividend_yield'),
                        data.get('beta'),
                        ticker
                    ))
            return True
        except Exception as e:
            logger.error(f"{ticker}: Failed to update fundamentals - {e}")
            return False


def test_finviz():
    """Test Finviz data fetching."""
    fetcher = FinvizDataFetcher()

    tickers = ['AAPL', 'MSFT', 'NVDA']

    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"{ticker}")
        print('='*50)

        # Get fundamentals
        fund = fetcher.get_fundamentals(ticker)
        if fund:
            print(f"Inst Own: {fund.get('inst_own')}%")
            print(f"Insider Own: {fund.get('insider_own')}%")
            print(f"Short Float: {fund.get('short_float')}%")
            print(f"P/E: {fund.get('pe')}")
            print(f"ROE: {fund.get('roe')}%")
            print(f"Target: ${fund.get('target_price')}")

        # Get institutional signal
        signal = fetcher.get_institutional_signal(ticker)
        print(f"\nInstitutional Signal: {signal['institutional_signal']}")
        print(f"Summary: {signal['summary']}")

        # Get ratings
        ratings = fetcher.get_analyst_ratings(ticker)
        print(f"\nRatings: {ratings.get('total_positive')}/{ratings.get('total_ratings')} positive")

        time.sleep(1)  # Rate limit


if __name__ == "__main__":
    test_finviz()