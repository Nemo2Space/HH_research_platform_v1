"""
SEC EDGAR Document Ingestion (Audit-Grade)
==========================================

Fetches and parses SEC filings (10-K, 10-Q, 8-K) from SEC EDGAR.

Features:
- Compliance-grade HTTP handling (rate limiting, backoff, caching)
- Proper User-Agent identification
- Text normalization for reproducibility
- Full provenance tracking (source_fetched_at_utc)
- Stores in rag.documents table

Usage:
    from src.rag.sec_ingestion import SECIngester

    ingester = SECIngester()
    ingester.ingest_ticker("MU", forms=["10-K", "10-Q"])
    ingester.ingest_universe(forms=["10-K"])

Author: HH Research Platform
"""

import os
import re
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from psycopg2.extras import Json
from src.db.connection import get_connection
from src.utils.logging import get_logger
from src.rag.rate_limiter import RateLimiter, RateLimiterConfig, create_sec_session

logger = get_logger(__name__)

# Try to import HTML parsing libraries
try:
    from bs4 import BeautifulSoup
    import html2text

    HTML_PARSING_AVAILABLE = True
except ImportError:
    HTML_PARSING_AVAILABLE = False
    logger.warning("beautifulsoup4/html2text not installed. Install with: pip install beautifulsoup4 html2text")


@dataclass
class SECFiling:
    """Represents a SEC filing with full provenance."""
    ticker: str
    cik: str
    accession_number: str
    form_type: str
    filing_date: datetime
    document_url: str
    fetched_at_utc: datetime
    raw_text: str
    raw_text_sha256: str
    char_count: int
    word_count: int
    period_label: str
    fiscal_year: Optional[int]
    fiscal_quarter: Optional[int]
    sections: Dict[str, str]


class TextNormalizer:
    """
    Deterministic text normalization.

    Rules:
    - Normalize line endings: \r\n and \r -> \n
    - Collapse excessive whitespace (but preserve paragraph breaks)
    - Strip leading/trailing whitespace
    - Remove null bytes and other control characters

    Note: After normalization, all offsets (char_start, char_end)
    reference the normalized text stored in raw_text.
    """

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text for consistent storage.

        This is the ONLY place normalization happens.
        All downstream processing uses the normalized text.
        """
        if not text:
            return ""

        # Step 1: Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Step 2: Remove null bytes and control characters (except newline, tab)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # Step 3: Collapse excessive newlines (more than 2 -> 2)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Step 4: Collapse multiple spaces/tabs to single space (within lines)
        text = re.sub(r'[^\S\n]+', ' ', text)

        # Step 5: Remove trailing whitespace from each line
        text = '\n'.join(line.rstrip() for line in text.split('\n'))

        # Step 6: Strip leading/trailing whitespace from entire text
        text = text.strip()

        return text


class SECIngester:
    """
    Ingests SEC filings into the RAG system.

    Uses compliance-grade HTTP handling with rate limiting and caching.
    """

    # Section patterns for 10-K/10-Q
    SECTION_PATTERNS = {
        '10-K': {
            'business': r'(?:ITEM\s*1[.\s]*[-–]?\s*)?BUSINESS(?!\s*RISK)',
            'risk_factors': r'(?:ITEM\s*1A[.\s]*[-–]?\s*)?RISK\s*FACTORS',
            'properties': r'(?:ITEM\s*2[.\s]*[-–]?\s*)?PROPERTIES',
            'legal': r'(?:ITEM\s*3[.\s]*[-–]?\s*)?LEGAL\s*PROCEEDINGS',
            'mda': r'(?:ITEM\s*7[.\s]*[-–]?\s*)?MANAGEMENT[\'\']?S?\s*DISCUSSION',
            'financials': r'(?:ITEM\s*8[.\s]*[-–]?\s*)?FINANCIAL\s*STATEMENTS',
        },
        '10-Q': {
            'financials': r'(?:ITEM\s*1[.\s]*[-–]?\s*)?FINANCIAL\s*STATEMENTS',
            'mda': r'(?:ITEM\s*2[.\s]*[-–]?\s*)?MANAGEMENT[\'\']?S?\s*DISCUSSION',
            'risk_factors': r'(?:ITEM\s*1A[.\s]*[-–]?\s*)?RISK\s*FACTORS',
        },
    }

    def __init__(self, user_agent: str = None):
        """
        Initialize SEC ingester.

        Args:
            user_agent: SEC-compliant User-Agent string (required by SEC)
        """
        if not HTML_PARSING_AVAILABLE:
            raise ImportError(
                "beautifulsoup4 and html2text required. "
                "Install with: pip install beautifulsoup4 html2text"
            )

        # Create session with rate limiter
        self.session, self.limiter = create_sec_session(user_agent)

        # HTML to text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False
        self.html_converter.body_width = 0  # No line wrapping

        # Text normalizer
        self.normalizer = TextNormalizer()

        # CIK cache
        self._cik_cache: Dict[str, str] = {}

    def get_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a ticker.

        Uses cached company tickers list from SEC.
        """
        ticker_upper = ticker.upper()

        if ticker_upper in self._cik_cache:
            return self._cik_cache[ticker_upper]

        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            response = self.limiter.get(
                self.session, url,
                ttl_seconds=86400  # 24 hour cache for company list
            )

            data = response.json()
            for item in data.values():
                if item.get('ticker', '').upper() == ticker_upper:
                    cik = str(item['cik_str']).zfill(10)
                    self._cik_cache[ticker_upper] = cik
                    return cik

            logger.warning(f"{ticker}: CIK not found in SEC database")
            return None

        except Exception as e:
            logger.error(f"Error fetching CIK for {ticker}: {e}")
            return None

    def get_filings_list(
            self,
            ticker: str,
            forms: List[str] = None,
            limit: int = 10
    ) -> List[Dict]:
        """
        Get list of recent filings for a ticker.

        Args:
            ticker: Stock ticker
            forms: List of form types (e.g., ['10-K', '10-Q'])
            limit: Maximum filings to return

        Returns:
            List of filing metadata dicts
        """
        if forms is None:
            forms = ['10-K', '10-Q']

        cik = self.get_cik(ticker)
        if not cik:
            return []

        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = self.limiter.get(
                self.session, url,
                ttl_seconds=86400  # 24 hour cache for filing list
            )

            data = response.json()

            filings = []
            recent = data.get('filings', {}).get('recent', {})

            if not recent:
                logger.warning(f"{ticker}: No recent filings found")
                return []

            form_types = recent.get('form', [])
            accession_numbers = recent.get('accessionNumber', [])
            filing_dates = recent.get('filingDate', [])
            primary_docs = recent.get('primaryDocument', [])

            for i, form_type in enumerate(form_types):
                if form_type in forms:
                    accession = accession_numbers[i].replace('-', '')

                    filing = {
                        'ticker': ticker,
                        'cik': cik,
                        'form_type': form_type,
                        'accession_number': accession_numbers[i],
                        'filing_date': filing_dates[i],
                        'primary_document': primary_docs[i],
                        'document_url': (
                            f"https://www.sec.gov/Archives/edgar/data/"
                            f"{cik.lstrip('0')}/{accession}/{primary_docs[i]}"
                        )
                    }
                    filings.append(filing)

                    if len(filings) >= limit:
                        break

            logger.info(f"{ticker}: Found {len(filings)} filings ({', '.join(forms)})")
            return filings

        except Exception as e:
            logger.error(f"Error fetching filings for {ticker}: {e}")
            return []

    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML to plain text."""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove unwanted tags
        for tag in soup(['script', 'style', 'meta', 'link', 'head']):
            tag.decompose()

        # Convert to text
        text = self.html_converter.handle(str(soup))

        return text

    def _clean_filing_text(self, text: str) -> str:
        """Remove filing-specific boilerplate."""
        # Remove page numbers
        text = re.sub(r'\n\s*-?\s*\d+\s*-?\s*\n', '\n', text)
        text = re.sub(r'\nPage\s+\d+\s*(?:of\s+\d+)?\n', '\n', text, flags=re.IGNORECASE)

        # Remove table of contents markers
        text = re.sub(r'\n\s*Table of Contents\s*\n', '\n', text, flags=re.IGNORECASE)

        return text

    def fetch_filing_document(self, filing: Dict) -> Optional[Tuple[str, datetime]]:
        """
        Fetch the full text of a filing document.

        Returns:
            (raw_text, fetched_at_utc) or None if error
        """
        try:
            url = filing['document_url']
            fetched_at = datetime.utcnow()

            # Fetch with rate limiting and caching (filings are immutable)
            response = self.limiter.get(
                self.session, url,
                ttl_seconds=86400 * 365  # Cache forever (filings don't change)
            )

            content = response.text

            # Detect and convert HTML
            if '<html' in content.lower() or '<body' in content.lower():
                text = self._html_to_text(content)
            else:
                text = content

            # Clean filing-specific boilerplate
            text = self._clean_filing_text(text)

            # Normalize text (CRITICAL: this is the canonical form)
            text = self.normalizer.normalize(text)

            logger.debug(f"Fetched {filing['form_type']} for {filing['ticker']}: {len(text)} chars")
            return text, fetched_at

        except Exception as e:
            logger.error(f"Error fetching document {filing.get('document_url')}: {e}")
            return None

    def parse_sections(self, text: str, form_type: str) -> Dict[str, str]:
        """Parse document into sections."""
        sections = {}
        patterns = self.SECTION_PATTERNS.get(form_type, {})

        if not patterns:
            return {'full_text': text}

        # Find all section starts
        section_positions = []
        for section_name, pattern in patterns.items():
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            for match in matches:
                section_positions.append((match.start(), section_name))

        # Sort by position
        section_positions.sort(key=lambda x: x[0])

        # Extract sections
        for i, (start, name) in enumerate(section_positions):
            if i + 1 < len(section_positions):
                end = section_positions[i + 1][0]
            else:
                end = len(text)

            section_text = text[start:end].strip()

            # Limit section size
            if len(section_text) > 100000:
                section_text = section_text[:100000] + "\n\n[TRUNCATED]"

            sections[name] = section_text

        if not sections:
            sections['full_text'] = text[:200000] if len(text) > 200000 else text

        return sections

    def _determine_period(
            self,
            filing: Dict,
            text: str
    ) -> Tuple[Optional[int], Optional[int], str]:
        """Determine fiscal year, quarter, and period label."""
        form_type = filing['form_type']
        filing_date = datetime.strptime(filing['filing_date'], '%Y-%m-%d')

        # Try to extract from text
        year_match = re.search(
            r'(?:fiscal\s+)?year\s+ended?\s+\w+\s+\d+,?\s*(\d{4})',
            text, re.IGNORECASE
        )
        quarter_match = re.search(
            r'(?:quarter|three months)\s+ended?\s+(\w+)\s+(\d+),?\s*(\d{4})',
            text, re.IGNORECASE
        )

        fiscal_year = int(year_match.group(1)) if year_match else filing_date.year
        fiscal_quarter = None

        if form_type == '10-Q' and quarter_match:
            month_str = quarter_match.group(1).lower()
            month_map = {
                'march': 1, 'mar': 1,
                'june': 2, 'jun': 2,
                'september': 3, 'sep': 3, 'sept': 3,
                'december': 4, 'dec': 4,
            }
            fiscal_quarter = month_map.get(month_str)

        # Build period label
        if form_type == '10-K':
            period_label = f"FY{fiscal_year}"
        elif form_type == '10-Q' and fiscal_quarter:
            period_label = f"{fiscal_year}Q{fiscal_quarter}"
        else:
            period_label = f"{fiscal_year}"

        return fiscal_year, fiscal_quarter, period_label

    def save_document(
            self,
            filing: Dict,
            text: str,
            fetched_at: datetime,
            sections: Dict[str, str]
    ) -> Optional[int]:
        """
        Save document to database.

        Returns:
            doc_id if saved, None if already exists or error
        """
        # Calculate hash of normalized text
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Determine period
        fiscal_year, fiscal_quarter, period_label = self._determine_period(filing, text)

        # Calculate lengths
        char_count = len(text)
        word_count = len(text.split())

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if already exists by hash
                    cur.execute(
                        "SELECT doc_id FROM rag.documents WHERE raw_text_sha256 = %s",
                        (text_hash,)
                    )
                    existing = cur.fetchone()
                    if existing:
                        logger.debug(f"{filing['ticker']}: Document already exists (doc_id={existing[0]})")
                        return existing[0]

                    # Check if exists by natural key (different content, same filing)
                    cur.execute("""
                        SELECT doc_id FROM rag.documents 
                        WHERE ticker = %s AND doc_type = %s AND period_label = %s AND source = %s
                    """, (
                        filing['ticker'],
                        filing['form_type'].lower().replace('-', ''),
                        period_label,
                        'sec_edgar'
                    ))
                    existing = cur.fetchone()
                    if existing:
                        logger.warning(
                            f"{filing['ticker']}: Filing exists with different content "
                            f"(doc_id={existing[0]}). Skipping."
                        )
                        return existing[0]

                    # Insert document
                    cur.execute("""
                        INSERT INTO rag.documents (
                            ticker, doc_type, source, source_url, source_fetched_at_utc,
                            asof_ts_utc, period_label, fiscal_year, fiscal_quarter,
                            title, raw_text, raw_text_sha256, char_count, word_count,
                            section_count, metadata
                        ) VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            %s, %s
                        ) RETURNING doc_id
                    """, (
                        filing['ticker'],
                        filing['form_type'].lower().replace('-', ''),
                        'sec_edgar',
                        filing['document_url'],
                        fetched_at,
                        datetime.strptime(filing['filing_date'], '%Y-%m-%d'),
                        period_label,
                        fiscal_year,
                        fiscal_quarter,
                        f"{filing['ticker']} {filing['form_type']} {period_label}",
                        text,
                        text_hash,
                        char_count,
                        word_count,
                        len(sections),
                        Json({
                            'cik': filing['cik'],
                            'accession_number': filing['accession_number'],
                            'sections': list(sections.keys()),
                        })
                    ))

                    doc_id = cur.fetchone()[0]
                    conn.commit()

                    logger.info(
                        f"{filing['ticker']}: Saved {filing['form_type']} {period_label} "
                        f"(doc_id={doc_id}, {char_count:,} chars)"
                    )
                    return doc_id

        except Exception as e:
            logger.error(f"Error saving document for {filing['ticker']}: {e}")
            return None

    def ingest_filing(self, filing: Dict) -> Optional[int]:
        """Ingest a single filing."""
        result = self.fetch_filing_document(filing)
        if not result:
            return None

        text, fetched_at = result
        sections = self.parse_sections(text, filing['form_type'])
        doc_id = self.save_document(filing, text, fetched_at, sections)

        return doc_id

    def ingest_ticker(
            self,
            ticker: str,
            forms: List[str] = None,
            limit: int = 5
    ) -> List[int]:
        """
        Ingest recent filings for a ticker.

        Args:
            ticker: Stock ticker
            forms: List of form types (default: ['10-K', '10-Q'])
            limit: Max filings per form type

        Returns:
            List of doc_ids created
        """
        if forms is None:
            forms = ['10-K', '10-Q']

        logger.info(f"{ticker}: Ingesting SEC filings ({', '.join(forms)})")

        doc_ids = []
        filings = self.get_filings_list(ticker, forms, limit=limit * len(forms))

        for filing in filings:
            doc_id = self.ingest_filing(filing)
            if doc_id:
                doc_ids.append(doc_id)

        logger.info(f"{ticker}: Ingested {len(doc_ids)} filings")
        return doc_ids

    def ingest_universe(
            self,
            tickers: List[str] = None,
            forms: List[str] = None,
            limit_per_ticker: int = 3
    ) -> Dict:
        """
        Ingest filings for multiple tickers.

        Args:
            tickers: List of tickers (default: load from universe)
            forms: Form types to fetch
            limit_per_ticker: Max filings per ticker per form

        Returns:
            Summary dict
        """
        if tickers is None:
            from src.db.repository import Repository
            repo = Repository()
            tickers = repo.get_universe()

        if forms is None:
            forms = ['10-K', '10-Q']

        results = {
            'total_tickers': len(tickers),
            'successful': [],
            'failed': [],
            'total_docs': 0,
        }

        logger.info(f"Ingesting SEC filings for {len(tickers)} tickers")

        for ticker in tickers:
            try:
                doc_ids = self.ingest_ticker(ticker, forms, limit=limit_per_ticker)
                if doc_ids:
                    results['successful'].append(ticker)
                    results['total_docs'] += len(doc_ids)
                else:
                    results['failed'].append(ticker)

            except Exception as e:
                logger.error(f"{ticker}: Ingestion failed - {e}")
                results['failed'].append(ticker)

        # Log rate limiter stats
        stats = self.limiter.get_stats()
        logger.info(
            f"SEC ingestion complete: {len(results['successful'])} success, "
            f"{len(results['failed'])} failed, {results['total_docs']} docs. "
            f"HTTP stats: {stats}"
        )

        return results


# Convenience functions
def ingest_sec_filings(ticker: str, forms: List[str] = None) -> List[int]:
    """Quick function to ingest SEC filings for a ticker."""
    ingester = SECIngester()
    return ingester.ingest_ticker(ticker, forms)


def ingest_sec_universe(tickers: List[str] = None) -> Dict:
    """Quick function to ingest SEC filings for universe."""
    ingester = SECIngester()
    return ingester.ingest_universe(tickers)


if __name__ == "__main__":
    # Test with a few tickers
    ingester = SECIngester()

    print("Testing SEC ingestion...")
    print(f"Rate limiter config: {ingester.limiter.config}")

    # Test CIK lookup
    cik = ingester.get_cik("AAPL")
    print(f"AAPL CIK: {cik}")

    # Test filing list
    filings = ingester.get_filings_list("AAPL", forms=['10-K'], limit=1)
    if filings:
        print(f"Found filing: {filings[0]['form_type']} from {filings[0]['filing_date']}")

    print(f"\nRate limiter stats: {ingester.limiter.get_stats()}")