"""
Form 4 Insider Transaction Tracker

Tracks insider buying/selling with only 2-day lag.
CEOs, CFOs, Directors buying their own stock = strong bullish signal.

Data Source: SEC EDGAR (free)
Filing Deadline: 2 business days after transaction

Signals:
- Cluster buying (multiple insiders) = very bullish
- CEO large purchase = bullish
- Insider selling after lockup = neutral (often planned)
- CFO dumping = bearish

Author: Alpha Research Platform
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import xml.etree.ElementTree as ET
import re
import time
import os

from src.utils.logging import get_logger

logger = get_logger(__name__)


# SEC requires user agent
SEC_USER_AGENT = os.getenv('SEC_USER_AGENT', 'AlphaResearch contact@example.com')
SEC_HEADERS = {
    'User-Agent': SEC_USER_AGENT,
    'Accept-Encoding': 'gzip, deflate',
}


class TransactionType(Enum):
    """Insider transaction types."""
    PURCHASE = "PURCHASE"           # Open market buy
    SALE = "SALE"                   # Open market sell
    OPTION_EXERCISE = "OPTION_EXERCISE"  # Exercise options
    GIFT = "GIFT"                   # Gift/donation
    AWARD = "AWARD"                 # Stock award/grant
    OTHER = "OTHER"


class InsiderRole(Enum):
    """Insider roles - higher = more significant."""
    CEO = "CEO"
    CFO = "CFO"
    COO = "COO"
    PRESIDENT = "PRESIDENT"
    CHAIRMAN = "CHAIRMAN"
    DIRECTOR = "DIRECTOR"
    VP = "VP"
    OFFICER = "OFFICER"
    TEN_PCT_OWNER = "10%_OWNER"
    OTHER = "OTHER"


@dataclass
class InsiderTransaction:
    """Single insider transaction."""
    ticker: str
    company_name: str

    # Insider info
    insider_name: str
    insider_role: InsiderRole
    insider_title: str  # Full title from filing

    # Transaction details
    transaction_type: TransactionType
    transaction_date: date
    filing_date: date

    # Shares & value
    shares: int
    price_per_share: float
    total_value: float

    # Post-transaction
    shares_owned_after: int = 0

    # Metadata
    form_type: str = "4"  # Form 4 or Form 5
    sec_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'company_name': self.company_name,
            'insider_name': self.insider_name,
            'insider_role': self.insider_role.value,
            'insider_title': self.insider_title,
            'transaction_type': self.transaction_type.value,
            'transaction_date': str(self.transaction_date),
            'filing_date': str(self.filing_date),
            'shares': self.shares,
            'price_per_share': self.price_per_share,
            'total_value': self.total_value,
            'shares_owned_after': self.shares_owned_after,
            'sec_url': self.sec_url,
        }


@dataclass
class InsiderSignal:
    """Aggregated insider signal for a ticker."""
    ticker: str
    as_of_date: date

    # Aggregate metrics (last 90 days)
    total_buys: int = 0
    total_sells: int = 0
    buy_value: float = 0
    sell_value: float = 0
    net_value: float = 0  # buy_value - sell_value

    # Unique insiders
    unique_buyers: int = 0
    unique_sellers: int = 0

    # Key insider activity
    ceo_bought: bool = False
    cfo_bought: bool = False
    ceo_sold: bool = False
    cfo_sold: bool = False

    # Cluster detection
    cluster_buying: bool = False  # 3+ insiders buying in 30 days
    cluster_selling: bool = False

    # Signal
    signal: str = "NEUTRAL"  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    signal_strength: int = 50

    # Recent transactions
    recent_transactions: List[InsiderTransaction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'total_buys': self.total_buys,
            'total_sells': self.total_sells,
            'buy_value': self.buy_value,
            'sell_value': self.sell_value,
            'net_value': self.net_value,
            'unique_buyers': self.unique_buyers,
            'unique_sellers': self.unique_sellers,
            'ceo_bought': self.ceo_bought,
            'cfo_bought': self.cfo_bought,
            'cluster_buying': self.cluster_buying,
            'cluster_selling': self.cluster_selling,
            'signal': self.signal,
            'signal_strength': self.signal_strength,
        }


class InsiderTracker:
    """
    Tracks Form 4 insider transactions from SEC EDGAR.
    """

    def __init__(self):
        self.base_url = "https://www.sec.gov"
        self.edgar_search = "https://efts.sec.gov/LATEST/search-index"
        self._cache = {}
        self._cache_duration = timedelta(hours=4)
        self._last_request = 0
        self._rate_limit = 0.15  # SEC allows 10 requests/sec, we do ~7

    def _rate_limited_get(self, url: str) -> Optional[requests.Response]:
        """Make rate-limited request to SEC."""
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)

        try:
            response = requests.get(url, headers=SEC_HEADERS, timeout=30)
            self._last_request = time.time()

            if response.status_code == 200:
                return response
            else:
                logger.warning(f"SEC request failed: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"SEC request error: {e}")
            return None

    def get_insider_transactions(self, ticker: str,
                                  days_back: int = 90) -> List[InsiderTransaction]:
        """
        Get insider transactions for a ticker.

        Args:
            ticker: Stock symbol
            days_back: Days of history to fetch

        Returns:
            List of InsiderTransaction objects
        """
        # Check cache
        cache_key = f"{ticker}_{days_back}"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_duration:
                return cached_data

        transactions = []

        try:
            # Get CIK for ticker
            cik = self._get_cik(ticker)
            if not cik:
                logger.warning(f"{ticker}: CIK not found")
                return transactions

            # Get recent Form 4 filings
            filings = self._get_form4_filings(cik, days_back)

            for filing in filings:
                try:
                    txns = self._parse_form4(filing, ticker)
                    transactions.extend(txns)
                except Exception as e:
                    logger.debug(f"Error parsing filing: {e}")
                    continue

            # Sort by date descending
            transactions.sort(key=lambda x: x.transaction_date, reverse=True)

            # Cache results
            self._cache[cache_key] = (datetime.now(), transactions)

        except Exception as e:
            logger.error(f"{ticker}: Error fetching insider transactions: {e}")

        return transactions

    def get_insider_signal(self, ticker: str) -> InsiderSignal:
        """
        Get aggregated insider signal for a ticker.

        Args:
            ticker: Stock symbol

        Returns:
            InsiderSignal with buy/sell analysis
        """
        signal = InsiderSignal(
            ticker=ticker,
            as_of_date=date.today(),
        )

        # Get transactions
        transactions = self.get_insider_transactions(ticker, days_back=90)

        if not transactions:
            return signal

        signal.recent_transactions = transactions[:10]  # Keep top 10

        # Analyze transactions
        buyers = set()
        sellers = set()

        thirty_days_ago = date.today() - timedelta(days=30)
        recent_buyers = set()
        recent_sellers = set()

        for txn in transactions:
            if txn.transaction_type == TransactionType.PURCHASE:
                signal.total_buys += 1
                signal.buy_value += txn.total_value
                buyers.add(txn.insider_name)

                if txn.transaction_date >= thirty_days_ago:
                    recent_buyers.add(txn.insider_name)

                # Check key insiders
                if txn.insider_role == InsiderRole.CEO:
                    signal.ceo_bought = True
                elif txn.insider_role == InsiderRole.CFO:
                    signal.cfo_bought = True

            elif txn.transaction_type == TransactionType.SALE:
                signal.total_sells += 1
                signal.sell_value += txn.total_value
                sellers.add(txn.insider_name)

                if txn.transaction_date >= thirty_days_ago:
                    recent_sellers.add(txn.insider_name)

                if txn.insider_role == InsiderRole.CEO:
                    signal.ceo_sold = True
                elif txn.insider_role == InsiderRole.CFO:
                    signal.cfo_sold = True

        signal.unique_buyers = len(buyers)
        signal.unique_sellers = len(sellers)
        signal.net_value = signal.buy_value - signal.sell_value

        # Cluster detection
        signal.cluster_buying = len(recent_buyers) >= 3
        signal.cluster_selling = len(recent_sellers) >= 3

        # Generate signal
        signal.signal, signal.signal_strength = self._calculate_signal(signal)

        return signal

    def _calculate_signal(self, signal: InsiderSignal) -> Tuple[str, int]:
        """Calculate overall insider signal."""
        score = 50

        # CEO/CFO buying is very bullish
        if signal.ceo_bought:
            score += 20
        if signal.cfo_bought:
            score += 15

        # Cluster buying is bullish
        if signal.cluster_buying:
            score += 15

        # Net buying vs selling
        if signal.net_value > 1000000:  # >$1M net buying
            score += 10
        elif signal.net_value > 100000:
            score += 5
        elif signal.net_value < -1000000:  # >$1M net selling
            score -= 10
        elif signal.net_value < -100000:
            score -= 5

        # Multiple buyers
        if signal.unique_buyers >= 3:
            score += 10
        elif signal.unique_buyers >= 2:
            score += 5

        # CEO selling is concerning
        if signal.ceo_sold and signal.ceo_bought:
            pass  # Mixed - ignore
        elif signal.ceo_sold:
            score -= 15

        # Cluster selling is bearish
        if signal.cluster_selling:
            score -= 15

        # Clamp score
        score = max(0, min(100, score))

        # Determine signal
        if score >= 75:
            return "STRONG_BUY", score
        elif score >= 60:
            return "BUY", score
        elif score <= 25:
            return "STRONG_SELL", score
        elif score <= 40:
            return "SELL", score
        else:
            return "NEUTRAL", score

    def _get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number for ticker."""
        try:
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=4&dateb=&owner=include&count=1&output=atom"
            response = self._rate_limited_get(url)

            if response and response.text:
                # Extract CIK from response
                match = re.search(r'CIK=(\d+)', response.text)
                if match:
                    return match.group(1).zfill(10)

            # Alternative: Try company tickers JSON
            tickers_url = "https://www.sec.gov/files/company_tickers.json"
            response = self._rate_limited_get(tickers_url)

            if response:
                data = response.json()
                for entry in data.values():
                    if entry.get('ticker', '').upper() == ticker.upper():
                        return str(entry.get('cik_str', '')).zfill(10)

        except Exception as e:
            logger.debug(f"CIK lookup error: {e}")

        return None

    def _get_form4_filings(self, cik: str, days_back: int) -> List[Dict]:
        """Get list of Form 4 filings for a CIK."""
        filings = []

        try:
            # Use SEC submissions API
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = self._rate_limited_get(url)

            if not response:
                return filings

            data = response.json()
            recent = data.get('filings', {}).get('recent', {})

            forms = recent.get('form', [])
            dates = recent.get('filingDate', [])
            accessions = recent.get('accessionNumber', [])

            cutoff = date.today() - timedelta(days=days_back)

            for i, form in enumerate(forms):
                if form in ['4', '4/A']:
                    filing_date = datetime.strptime(dates[i], '%Y-%m-%d').date()

                    if filing_date >= cutoff:
                        accession = accessions[i].replace('-', '')
                        filings.append({
                            'cik': cik,
                            'accession': accession,
                            'form': form,
                            'filing_date': filing_date,
                        })

        except Exception as e:
            logger.debug(f"Error getting filings: {e}")

        return filings[:50]  # Limit to 50 most recent

    def _parse_form4(self, filing: Dict, ticker: str) -> List[InsiderTransaction]:
        """Parse a Form 4 filing."""
        transactions = []

        try:
            cik = filing['cik']
            accession = filing['accession']

            # Get filing documents
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}"

            # Find XML file
            index_url = f"{url}/index.json"
            response = self._rate_limited_get(index_url)

            if not response:
                return transactions

            index_data = response.json()
            xml_file = None

            for item in index_data.get('directory', {}).get('item', []):
                name = item.get('name', '')
                if name.endswith('.xml') and 'primary_doc' not in name.lower():
                    xml_file = name
                    break

            if not xml_file:
                return transactions

            # Get and parse XML
            xml_url = f"{url}/{xml_file}"
            response = self._rate_limited_get(xml_url)

            if not response:
                return transactions

            root = ET.fromstring(response.content)

            # Get company name
            company_name = ""
            issuer = root.find('.//issuer')
            if issuer is not None:
                name_elem = issuer.find('issuerName')
                if name_elem is not None:
                    company_name = name_elem.text or ""

            # Get insider info
            insider_name = ""
            insider_title = ""

            owner = root.find('.//reportingOwner')
            if owner is not None:
                name_elem = owner.find('.//rptOwnerName')
                if name_elem is not None:
                    insider_name = name_elem.text or ""

                title_elem = owner.find('.//officerTitle')
                if title_elem is not None:
                    insider_title = title_elem.text or ""

                # Check relationship
                relationship = owner.find('.//reportingOwnerRelationship')
                if relationship is not None:
                    is_director = relationship.find('isDirector')
                    is_officer = relationship.find('isOfficer')
                    is_ten_pct = relationship.find('isTenPercentOwner')

            insider_role = self._parse_role(insider_title)

            # Parse transactions
            for txn in root.findall('.//nonDerivativeTransaction'):
                try:
                    trans = self._parse_transaction_element(
                        txn, ticker, company_name, insider_name,
                        insider_role, insider_title, filing
                    )
                    if trans:
                        transactions.append(trans)
                except Exception as e:
                    logger.debug(f"Transaction parse error: {e}")

        except Exception as e:
            logger.debug(f"Form 4 parse error: {e}")

        return transactions

    def _parse_transaction_element(self, elem, ticker: str, company_name: str,
                                    insider_name: str, insider_role: InsiderRole,
                                    insider_title: str, filing: Dict) -> Optional[InsiderTransaction]:
        """Parse a single transaction element."""
        try:
            # Transaction date
            date_elem = elem.find('.//transactionDate/value')
            if date_elem is None:
                return None
            txn_date = datetime.strptime(date_elem.text, '%Y-%m-%d').date()

            # Transaction code
            code_elem = elem.find('.//transactionCoding/transactionCode')
            code = code_elem.text if code_elem is not None else ""

            # Map code to type
            txn_type = self._map_transaction_code(code)
            if txn_type == TransactionType.OTHER:
                return None  # Skip non-meaningful transactions

            # Shares
            shares_elem = elem.find('.//transactionAmounts/transactionShares/value')
            shares = int(float(shares_elem.text)) if shares_elem is not None else 0

            # Price
            price_elem = elem.find('.//transactionAmounts/transactionPricePerShare/value')
            price = float(price_elem.text) if price_elem is not None and price_elem.text else 0

            # Acquisition or disposition
            acq_elem = elem.find('.//transactionAmounts/transactionAcquiredDisposedCode/value')
            is_acquisition = acq_elem is not None and acq_elem.text == 'A'

            # Adjust type based on A/D code
            if txn_type == TransactionType.SALE and is_acquisition:
                txn_type = TransactionType.PURCHASE
            elif txn_type == TransactionType.PURCHASE and not is_acquisition:
                txn_type = TransactionType.SALE

            # Shares owned after
            owned_elem = elem.find('.//postTransactionAmounts/sharesOwnedFollowingTransaction/value')
            shares_after = int(float(owned_elem.text)) if owned_elem is not None else 0

            # Calculate total value
            total_value = shares * price

            if shares == 0 or price == 0:
                return None  # Skip transactions without meaningful data

            return InsiderTransaction(
                ticker=ticker,
                company_name=company_name,
                insider_name=insider_name,
                insider_role=insider_role,
                insider_title=insider_title,
                transaction_type=txn_type,
                transaction_date=txn_date,
                filing_date=filing['filing_date'],
                shares=shares,
                price_per_share=price,
                total_value=total_value,
                shares_owned_after=shares_after,
                sec_url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=4",
            )

        except Exception as e:
            logger.debug(f"Transaction element parse error: {e}")
            return None

    def _parse_role(self, title: str) -> InsiderRole:
        """Parse insider role from title."""
        title_upper = (title or "").upper()

        if "CEO" in title_upper or "CHIEF EXECUTIVE" in title_upper:
            return InsiderRole.CEO
        elif "CFO" in title_upper or "CHIEF FINANCIAL" in title_upper:
            return InsiderRole.CFO
        elif "COO" in title_upper or "CHIEF OPERATING" in title_upper:
            return InsiderRole.COO
        elif "PRESIDENT" in title_upper:
            return InsiderRole.PRESIDENT
        elif "CHAIRMAN" in title_upper:
            return InsiderRole.CHAIRMAN
        elif "DIRECTOR" in title_upper:
            return InsiderRole.DIRECTOR
        elif "VP" in title_upper or "VICE PRESIDENT" in title_upper:
            return InsiderRole.VP
        elif "OFFICER" in title_upper:
            return InsiderRole.OFFICER
        else:
            return InsiderRole.OTHER

    def _map_transaction_code(self, code: str) -> TransactionType:
        """Map SEC transaction code to type."""
        code = code.upper() if code else ""

        if code == 'P':
            return TransactionType.PURCHASE
        elif code == 'S':
            return TransactionType.SALE
        elif code in ['M', 'C']:
            return TransactionType.OPTION_EXERCISE
        elif code == 'G':
            return TransactionType.GIFT
        elif code == 'A':
            return TransactionType.AWARD
        else:
            return TransactionType.OTHER


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_tracker = None

def get_insider_tracker() -> InsiderTracker:
    """Get singleton tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = InsiderTracker()
    return _tracker


def get_insider_signal(ticker: str) -> InsiderSignal:
    """
    Get insider signal for a ticker.

    Usage:
        signal = get_insider_signal('AAPL')
        print(f"Signal: {signal.signal}")
        print(f"CEO bought: {signal.ceo_bought}")
        print(f"Net value: ${signal.net_value:,.0f}")
    """
    tracker = get_insider_tracker()
    return tracker.get_insider_signal(ticker)


def get_insider_transactions(ticker: str, days_back: int = 90) -> List[InsiderTransaction]:
    """
    Get raw insider transactions.

    Usage:
        txns = get_insider_transactions('AAPL', days_back=30)
        for t in txns:
            print(f"{t.insider_name}: {t.transaction_type.value} {t.shares} @ ${t.price_per_share}")
    """
    tracker = get_insider_tracker()
    return tracker.get_insider_transactions(ticker, days_back)


def get_insider_summary_table(tickers: List[str]) -> pd.DataFrame:
    """
    Get insider signals for multiple tickers as DataFrame.

    Usage:
        df = get_insider_summary_table(['AAPL', 'MSFT', 'NVDA'])
        print(df)
    """
    tracker = get_insider_tracker()

    data = []
    for ticker in tickers:
        try:
            signal = tracker.get_insider_signal(ticker)
            data.append({
                'Ticker': ticker,
                'Signal': signal.signal,
                'Strength': signal.signal_strength,
                'Buys': signal.total_buys,
                'Sells': signal.total_sells,
                'Net Value': f"${signal.net_value:,.0f}",
                'CEO Buy': '✅' if signal.ceo_bought else '',
                'CFO Buy': '✅' if signal.cfo_bought else '',
                'Cluster': '🔥' if signal.cluster_buying else '',
            })
        except Exception as e:
            logger.debug(f"{ticker}: Insider signal error: {e}")

    return pd.DataFrame(data)


def screen_for_insider_buying(tickers: List[str], min_strength: int = 60) -> List[InsiderSignal]:
    """
    Screen for stocks with insider buying.

    Usage:
        signals = screen_for_insider_buying(universe, min_strength=70)
        for s in signals:
            print(f"{s.ticker}: {s.signal} - CEO bought: {s.ceo_bought}")
    """
    tracker = get_insider_tracker()

    results = []
    for ticker in tickers:
        try:
            signal = tracker.get_insider_signal(ticker)
            if signal.signal_strength >= min_strength:
                results.append(signal)
        except Exception:
            continue

    results.sort(key=lambda x: x.signal_strength, reverse=True)
    return results


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Insider Tracker...")

    # Test single ticker
    signal = get_insider_signal('AAPL')

    print(f"\nAAPL Insider Signal:")
    print(f"  Signal: {signal.signal} (strength: {signal.signal_strength})")
    print(f"  Total Buys: {signal.total_buys}")
    print(f"  Total Sells: {signal.total_sells}")
    print(f"  Net Value: ${signal.net_value:,.0f}")
    print(f"  CEO Bought: {signal.ceo_bought}")
    print(f"  Cluster Buying: {signal.cluster_buying}")

    if signal.recent_transactions:
        print(f"\n  Recent Transactions:")
        for t in signal.recent_transactions[:5]:
            print(f"    {t.transaction_date}: {t.insider_name} ({t.insider_role.value}) "
                  f"{t.transaction_type.value} {t.shares:,} @ ${t.price_per_share:.2f}")