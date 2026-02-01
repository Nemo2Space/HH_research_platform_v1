"""
Alpha Platform - SEC Filing Fetcher

Fetches SEC filings (10-K, 10-Q, 8-K) from EDGAR.
"""

import os
import re
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.db.connection import get_connection
from src.db.repository import Repository
from src.utils.logging import get_logger

logger = get_logger(__name__)

# SEC EDGAR base URL
SEC_BASE_URL = "https://data.sec.gov"
SEC_SUBMISSIONS_URL = f"{SEC_BASE_URL}/submissions"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

# Required headers for SEC API
SEC_HEADERS = {
    "User-Agent": "AlphaPlatform/1.0 (contact@example.com)",
    "Accept-Encoding": "gzip, deflate",
}

# Filing types we care about
FILING_TYPES = ["10-K", "10-Q", "8-K"]


@dataclass
class SECFiling:
    """SEC filing data."""
    ticker: str
    cik: str
    form_type: str
    filing_date: str
    accession_number: str
    document_url: str
    description: str = ""
    content: str = ""


class SECFilingFetcher:
    """Fetches and parses SEC filings."""

    def __init__(self, repository: Optional[Repository] = None):
        self.repo = repository or Repository()
        self.session = requests.Session()
        self.session.headers.update(SEC_HEADERS)

        # CIK cache
        self._cik_cache: Dict[str, str] = {}

    def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK number for a ticker."""
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]

        try:
            # Use SEC company tickers JSON
            url = "https://www.sec.gov/files/company_tickers.json"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Search for ticker
            for entry in data.values():
                if entry.get('ticker', '').upper() == ticker.upper():
                    cik = str(entry.get('cik_str', '')).zfill(10)
                    self._cik_cache[ticker] = cik
                    return cik

            logger.warning(f"{ticker}: CIK not found")
            return None

        except Exception as e:
            logger.error(f"{ticker}: Error getting CIK - {e}")
            return None

    def get_recent_filings(self, ticker: str, form_types: List[str] = None,
                           limit: int = 10) -> List[SECFiling]:
        """
        Get recent SEC filings for a ticker.

        Args:
            ticker: Stock ticker
            form_types: List of form types (default: 10-K, 10-Q, 8-K)
            limit: Maximum filings to return

        Returns:
            List of SECFiling objects
        """
        form_types = form_types or FILING_TYPES

        cik = self.get_cik(ticker)
        if not cik:
            return []

        try:
            # Get submissions
            url = f"{SEC_SUBMISSIONS_URL}/CIK{cik}.json"
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            data = response.json()

            filings = []
            recent = data.get('filings', {}).get('recent', {})

            forms = recent.get('form', [])
            dates = recent.get('filingDate', [])
            accessions = recent.get('accessionNumber', [])
            docs = recent.get('primaryDocument', [])
            descriptions = recent.get('primaryDocDescription', [])

            for i in range(min(len(forms), 100)):  # Check first 100
                form = forms[i]

                if form not in form_types:
                    continue

                accession = accessions[i].replace('-', '')
                doc_url = f"{SEC_ARCHIVES_URL}/{cik.lstrip('0')}/{accession}/{docs[i]}"

                filing = SECFiling(
                    ticker=ticker,
                    cik=cik,
                    form_type=form,
                    filing_date=dates[i],
                    accession_number=accessions[i],
                    document_url=doc_url,
                    description=descriptions[i] if i < len(descriptions) else ""
                )

                filings.append(filing)

                if len(filings) >= limit:
                    break

            logger.info(f"{ticker}: Found {len(filings)} filings")
            return filings

        except Exception as e:
            logger.error(f"{ticker}: Error fetching filings - {e}")
            return []

    def fetch_filing_content(self, filing: SECFiling, max_chars: int = 500000) -> str:
        """
        Fetch the actual content of a filing.

        Args:
            filing: SECFiling object
            max_chars: Maximum characters to fetch

        Returns:
            Cleaned text content
        """
        try:
            response = self.session.get(filing.document_url, timeout=30)
            response.raise_for_status()

            content = response.text

            # Clean HTML
            content = self._clean_html(content)

            # Truncate if too long
            if len(content) > max_chars:
                content = content[:max_chars]

            logger.info(f"{filing.ticker}: Fetched {len(content)} chars from {filing.form_type}")
            return content

        except Exception as e:
            logger.error(f"{filing.ticker}: Error fetching content - {e}")
            return ""

    def _clean_html(self, html: str) -> str:
        """Clean HTML content to plain text."""
        # Remove script and style tags
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags
        html = re.sub(r'<[^>]+>', ' ', html)

        # Decode entities
        html = html.replace('&nbsp;', ' ')
        html = html.replace('&amp;', '&')
        html = html.replace('&lt;', '<')
        html = html.replace('&gt;', '>')
        html = html.replace('&quot;', '"')

        # Clean whitespace
        html = re.sub(r'\s+', ' ', html)
        html = html.strip()

        return html

    def save_filing(self, filing: SECFiling) -> bool:
        """Save filing to database."""
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO sec_filings (
                            ticker, cik, form_type, filing_date, accession_number,
                            document_url, description, content, content_length
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, accession_number) DO UPDATE SET
                            content = EXCLUDED.content,
                            content_length = EXCLUDED.content_length
                    """, (
                        filing.ticker,
                        filing.cik,
                        filing.form_type,
                        filing.filing_date,
                        filing.accession_number,
                        filing.document_url,
                        filing.description,
                        filing.content,
                        len(filing.content)
                    ))
            return True
        except Exception as e:
            logger.error(f"Error saving filing: {e}")
            return False

    def fetch_and_save(self, ticker: str, form_types: List[str] = None,
                       limit: int = 3) -> int:
        """
        Fetch recent filings and save to database.

        Returns:
            Number of filings saved
        """
        filings = self.get_recent_filings(ticker, form_types, limit)

        saved = 0
        for filing in filings:
            # Check if already in database
            existing = self._check_existing(filing)
            if existing:
                logger.debug(f"{ticker}: Filing {filing.accession_number} already exists")
                continue

            # Fetch content
            filing.content = self.fetch_filing_content(filing)

            if filing.content:
                if self.save_filing(filing):
                    saved += 1

            # Rate limiting
            time.sleep(0.2)

        return saved

    def _check_existing(self, filing: SECFiling) -> bool:
        """Check if filing already exists in database."""
        try:
            import pandas as pd
            query = """
                SELECT id FROM sec_filings 
                WHERE ticker = %(ticker)s AND accession_number = %(accession)s
            """
            df = pd.read_sql(query, self.repo.engine, params={
                "ticker": filing.ticker,
                "accession": filing.accession_number
            })
            return len(df) > 0
        except:
            return False

    def fetch_universe(self, form_types: List[str] = None, limit_per_ticker: int = 2,
                       delay: float = 0.5, progress_callback=None) -> Dict[str, int]:
        """Fetch filings for all tickers in universe."""
        tickers = self.repo.get_universe()
        total = len(tickers)

        results = {}

        for i, ticker in enumerate(tickers):
            if progress_callback:
                progress_callback(i + 1, total, ticker)

            count = self.fetch_and_save(ticker, form_types, limit_per_ticker)
            results[ticker] = count

            if delay > 0:
                time.sleep(delay)

        total_saved = sum(results.values())
        logger.info(f"Fetched {total_saved} filings for {total} tickers")

        return results