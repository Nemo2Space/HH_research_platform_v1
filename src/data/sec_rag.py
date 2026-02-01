"""
Alpha Platform - SEC Filing RAG

Chunks SEC filings and provides retrieval for agent analysis.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from src.db.connection import get_connection
from src.db.repository import Repository
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Chunk configuration
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200


@dataclass
class SECChunk:
    """A chunk of SEC filing text."""
    filing_id: int
    ticker: str
    form_type: str
    chunk_index: int
    content: str
    section: str = ""
    embedding: Optional[List[float]] = None


class SECChunker:
    """Chunks SEC filings for RAG retrieval."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                 repository: Optional[Repository] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.repo = repository or Repository()

    def detect_section(self, text: str) -> str:
        """Detect SEC filing section from text content."""
        text_lower = text[:500].lower()

        sections = {
            'risk factors': 'RISK_FACTORS',
            'item 1a': 'RISK_FACTORS',
            'management discussion': 'MD&A',
            'item 7': 'MD&A',
            'business': 'BUSINESS',
            'item 1': 'BUSINESS',
            'financial statements': 'FINANCIALS',
            'item 8': 'FINANCIALS',
            'legal proceedings': 'LEGAL',
            'item 3': 'LEGAL',
            'executive compensation': 'COMPENSATION',
            'item 11': 'COMPENSATION',
        }

        for pattern, section in sections.items():
            if pattern in text_lower:
                return section

        return 'OTHER'

    def chunk_text(self, text: str, filing_id: int, ticker: str,
                   form_type: str) -> List[SECChunk]:
        """
        Split filing text into overlapping chunks.

        Args:
            text: Full filing text
            filing_id: Database ID of the filing
            ticker: Stock ticker
            form_type: Form type (10-K, 10-Q, 8-K)

        Returns:
            List of SECChunk objects
        """
        if not text:
            return []

        chunks = []

        # Split into sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        chunk_index = 0

        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    section = self.detect_section(current_chunk)
                    chunks.append(SECChunk(
                        filing_id=filing_id,
                        ticker=ticker,
                        form_type=form_type,
                        chunk_index=chunk_index,
                        content=current_chunk.strip(),
                        section=section
                    ))
                    chunk_index += 1

                    # Keep overlap
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence

        # Don't forget the last chunk
        if current_chunk.strip():
            section = self.detect_section(current_chunk)
            chunks.append(SECChunk(
                filing_id=filing_id,
                ticker=ticker,
                form_type=form_type,
                chunk_index=chunk_index,
                content=current_chunk.strip(),
                section=section
            ))

        return chunks

    def save_chunks(self, chunks: List[SECChunk]) -> int:
        """Save chunks to database."""
        saved = 0

        # Convert numpy types to Python native types
        def to_python(val):
            if val is None:
                return None
            try:
                import numpy as np
                if isinstance(val, (np.integer, np.int64, np.int32)):
                    return int(val)
                if isinstance(val, (np.floating, np.float64, np.float32)):
                    return float(val)
            except (ImportError, TypeError):
                pass
            return val

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    for chunk in chunks:
                        # Generate chunk hash for deduplication
                        chunk_hash = hashlib.md5(chunk.content.encode()).hexdigest()

                        cur.execute("""
                                    INSERT INTO sec_chunks (filing_id, ticker, form_type, chunk_index,
                                                            content, section, chunk_hash)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (filing_id, chunk_index) DO
                                    UPDATE SET
                                        content = EXCLUDED.content,
                                        section = EXCLUDED.section
                                    """, (
                                        to_python(chunk.filing_id),
                                        chunk.ticker,
                                        chunk.form_type,
                                        to_python(chunk.chunk_index),
                                        chunk.content,
                                        chunk.section,
                                        chunk_hash
                                    ))
                        saved += 1

            return saved

        except Exception as e:
            logger.error(f"Error saving chunks: {e}")
            return 0



    def process_filing(self, filing_id: int) -> int:
        """Process a single filing and create chunks."""
        # Get filing from database
        query = """
            SELECT id, ticker, form_type, content 
            FROM sec_filings 
            WHERE id = %(id)s
        """
        df = pd.read_sql(query, self.repo.engine, params={"id": filing_id})

        if len(df) == 0:
            return 0

        row = df.iloc[0]

        chunks = self.chunk_text(
            text=row['content'],
            filing_id=row['id'],
            ticker=row['ticker'],
            form_type=row['form_type']
        )

        saved = self.save_chunks(chunks)
        logger.info(f"Filing {filing_id}: Created {saved} chunks")

        return saved

    def process_all_filings(self, progress_callback=None) -> int:
        """Process all filings that haven't been chunked yet."""
        # Get filings without chunks
        query = """
            SELECT f.id, f.ticker, f.form_type
            FROM sec_filings f
            LEFT JOIN sec_chunks c ON f.id = c.filing_id
            WHERE c.id IS NULL AND f.content IS NOT NULL AND f.content != ''
        """
        df = pd.read_sql(query, self.repo.engine)

        total = len(df)
        total_chunks = 0

        for i, row in df.iterrows():
            if progress_callback:
                progress_callback(i + 1, total, row['ticker'])

            chunks = self.process_filing(row['id'])
            total_chunks += chunks

        logger.info(f"Processed {total} filings, created {total_chunks} chunks")
        return total_chunks


class SECRetriever:
    """Retrieves relevant SEC chunks for analysis."""

    def __init__(self, repository: Optional[Repository] = None):
        self.repo = repository or Repository()

    def search_chunks(self, ticker: str, query: str = None,
                      form_types: List[str] = None,
                      sections: List[str] = None,
                      limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relevant SEC chunks.

        Args:
            ticker: Stock ticker
            query: Optional keyword query
            form_types: Filter by form types
            sections: Filter by sections (RISK_FACTORS, MD&A, etc.)
            limit: Maximum results

        Returns:
            List of chunk dictionaries
        """
        conditions = ["c.ticker = %(ticker)s"]
        params = {"ticker": ticker, "limit": limit}

        if form_types:
            conditions.append("c.form_type = ANY(%(form_types)s)")
            params["form_types"] = form_types

        if sections:
            conditions.append("c.section = ANY(%(sections)s)")
            params["sections"] = sections

        # Simple keyword search
        if query:
            keywords = query.lower().split()
            for i, kw in enumerate(keywords[:5]):  # Max 5 keywords
                param_name = f"kw_{i}"
                conditions.append(f"LOWER(c.content) LIKE %({param_name})s")
                params[param_name] = f"%{kw}%"

        where_clause = " AND ".join(conditions)

        sql = f"""
            SELECT c.id, c.ticker, c.form_type, c.section, c.content,
                   f.filing_date, f.accession_number
            FROM sec_chunks c
            JOIN sec_filings f ON c.filing_id = f.id
            WHERE {where_clause}
            ORDER BY f.filing_date DESC, c.chunk_index
            LIMIT %(limit)s
        """

        try:
            df = pd.read_sql(sql, self.repo.engine, params=params)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def get_risk_factors(self, ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get risk factor chunks for a ticker."""
        return self.search_chunks(
            ticker=ticker,
            sections=['RISK_FACTORS'],
            limit=limit
        )

    def get_mda(self, ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get MD&A (Management Discussion & Analysis) chunks."""
        return self.search_chunks(
            ticker=ticker,
            sections=['MD&A'],
            limit=limit
        )

    def get_recent_8k(self, ticker: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get recent 8-K filing chunks."""
        return self.search_chunks(
            ticker=ticker,
            form_types=['8-K'],
            limit=limit
        )

    def get_context_for_analysis(self, ticker: str, max_chunks: int = 10) -> str:
        """
        Get SEC filing context for agent analysis.

        Returns formatted context string with key information.
        """
        context_parts = []

        # Get risk factors
        risks = self.get_risk_factors(ticker, limit=3)
        if risks:
            context_parts.append("=== RISK FACTORS ===")
            for chunk in risks:
                context_parts.append(chunk['content'][:800])

        # Get MD&A
        mda = self.get_mda(ticker, limit=3)
        if mda:
            context_parts.append("\n=== MANAGEMENT DISCUSSION ===")
            for chunk in mda:
                context_parts.append(chunk['content'][:800])

        # Get recent 8-K
        recent = self.get_recent_8k(ticker, limit=2)
        if recent:
            context_parts.append("\n=== RECENT 8-K FILINGS ===")
            for chunk in recent:
                context_parts.append(f"[{chunk['filing_date']}] {chunk['content'][:500]}")

        return "\n\n".join(context_parts) if context_parts else ""