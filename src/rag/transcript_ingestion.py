"""
Manual Transcript Upload & Ingestion
=====================================

Allows manual upload of earnings call transcripts when API sources
are not available or for testing purposes.

Features:
- Upload from file (txt, html, pdf)
- Paste raw transcript text
- Auto-detect sections (Prepared Remarks, Q&A)
- Parse speaker turns
- Store in documents table

Supported formats:
- Plain text (.txt)
- HTML (.html)
- PDF (.pdf) - requires pdfplumber

Usage:
    from src.rag.transcript_ingestion import TranscriptIngester

    ingester = TranscriptIngester()

    # From file
    doc_id = ingester.ingest_file("MU", "/path/to/transcript.txt", period="2025Q3")

    # From text
    doc_id = ingester.ingest_text("MU", raw_text, period="2025Q3")

    # Streamlit upload
    doc_id = ingester.ingest_uploaded_file("MU", uploaded_file, period="2025Q3")

Author: HH Research Platform
"""

import os
import re
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from src.db.connection import get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import optional dependencies
try:
    import pdfplumber

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.debug("pdfplumber not installed - PDF upload disabled")

try:
    from bs4 import BeautifulSoup
    import html2text

    HTML_AVAILABLE = True
except ImportError:
    HTML_AVAILABLE = False
    logger.debug("beautifulsoup4/html2text not installed - HTML parsing limited")


@dataclass
class TranscriptSection:
    """Represents a section of a transcript."""
    name: str  # 'Prepared Remarks', 'Q&A', 'Operator Instructions'
    text: str
    start_idx: int
    end_idx: int
    speakers: List[str] = field(default_factory=list)


@dataclass
class SpeakerTurn:
    """Represents a single speaker turn."""
    speaker: str
    role: str  # 'executive', 'analyst', 'operator'
    text: str
    section: str
    turn_index: int


class TranscriptIngester:
    """
    Ingests earnings call transcripts into the RAG system.

    Supports manual upload via file or text paste.
    """

    # Patterns for detecting sections
    SECTION_PATTERNS = {
        'prepared_remarks': [
            r'(?:^|\n)\s*(?:Prepared\s+Remarks|Opening\s+Remarks|Management\s+(?:Discussion|Commentary))\s*(?:\n|$)',
            r'(?:^|\n)\s*(?:Presentation|Corporate\s+Participants)\s*(?:\n|$)',
        ],
        'qa': [
            r'(?:^|\n)\s*(?:Question[s]?\s*(?:and|&)\s*Answer[s]?|Q\s*&\s*A\s+Session|Q&A)\s*(?:\n|$)',
            r'(?:^|\n)\s*(?:Analyst\s+Questions|Questions\s+from\s+(?:Analysts|the\s+Floor))\s*(?:\n|$)',
        ],
        'operator': [
            r'(?:^|\n)\s*Operator(?:\s+Instructions)?\s*(?:\n|$)',
        ],
    }

    # Patterns for detecting speakers
    SPEAKER_PATTERNS = [
        # "John Smith - CEO" or "John Smith, CEO"
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*[-–,]\s*([A-Za-z\s,]+?)(?:\n|$)',
        # "[John Smith]" or "(John Smith)"
        r'^\[([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\]',
        r'^\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\)',
        # "JOHN SMITH:" (all caps)
        r'^([A-Z][A-Z\s]+):',
    ]

    # Known executive titles
    EXECUTIVE_TITLES = [
        'ceo', 'chief executive', 'president',
        'cfo', 'chief financial', 'finance',
        'coo', 'chief operating', 'operations',
        'cto', 'chief technology', 'technology',
        'cmo', 'chief marketing', 'marketing',
        'evp', 'svp', 'vp', 'vice president',
        'director', 'head of', 'general manager',
        'treasurer', 'controller', 'ir ',
    ]

    # Known analyst indicators
    ANALYST_INDICATORS = [
        'analyst', 'research', 'securities', 'capital',
        'morgan', 'goldman', 'jpmorgan', 'citi', 'barclays',
        'wells fargo', 'bofa', 'credit suisse', 'ubs',
        'deutsche', 'hsbc', 'nomura', 'mizuho',
    ]

    def __init__(self):
        if HTML_AVAILABLE:
            self.html_converter = html2text.HTML2Text()
            self.html_converter.ignore_links = True
            self.html_converter.ignore_images = True
            self.html_converter.body_width = 0

    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

        elif suffix == '.html' or suffix == '.htm':
            if not HTML_AVAILABLE:
                raise ImportError("beautifulsoup4 and html2text required for HTML files")

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'html.parser')
            for tag in soup(['script', 'style', 'meta', 'link']):
                tag.decompose()

            return self.html_converter.handle(str(soup))

        elif suffix == '.pdf':
            if not PDF_AVAILABLE:
                raise ImportError("pdfplumber required for PDF files. Install with: pip install pdfplumber")

            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            return '\n\n'.join(text_parts)

        else:
            # Try reading as text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

    def _clean_text(self, text: str) -> str:
        """Clean up transcript text."""
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Remove page numbers
        text = re.sub(r'\n\s*-?\s*\d+\s*-?\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page\s+\d+\s*(?:of\s+\d+)?\s*\n', '\n', text, flags=re.IGNORECASE)

        # Remove copyright notices
        text = re.sub(r'\n.*(?:Copyright|©|All Rights Reserved).*\n', '\n', text, flags=re.IGNORECASE)

        return text.strip()

    def detect_sections(self, text: str) -> List[TranscriptSection]:
        """
        Detect major sections in transcript.

        Returns list of TranscriptSection objects.
        """
        sections = []
        section_positions = []

        # Find all section markers
        for section_name, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                    section_positions.append((match.start(), section_name, match.group().strip()))

        # Sort by position
        section_positions.sort(key=lambda x: x[0])

        # If no sections found, treat entire text as one section
        if not section_positions:
            sections.append(TranscriptSection(
                name='full_transcript',
                text=text,
                start_idx=0,
                end_idx=len(text)
            ))
            return sections

        # Extract sections
        for i, (start, name, header) in enumerate(section_positions):
            if i + 1 < len(section_positions):
                end = section_positions[i + 1][0]
            else:
                end = len(text)

            section_text = text[start:end].strip()

            # Map to standard names
            if 'qa' in name or 'question' in name.lower():
                std_name = 'Q&A'
            elif 'prepared' in name or 'presentation' in name.lower():
                std_name = 'Prepared Remarks'
            elif 'operator' in name:
                std_name = 'Operator'
            else:
                std_name = name.replace('_', ' ').title()

            sections.append(TranscriptSection(
                name=std_name,
                text=section_text,
                start_idx=start,
                end_idx=end
            ))

        return sections

    def _detect_speaker_role(self, speaker: str, title: str = '') -> str:
        """Determine if speaker is executive, analyst, or operator."""
        combined = f"{speaker} {title}".lower()

        if 'operator' in combined:
            return 'operator'

        for indicator in self.EXECUTIVE_TITLES:
            if indicator in combined:
                return 'executive'

        for indicator in self.ANALYST_INDICATORS:
            if indicator in combined:
                return 'analyst'

        # Default based on context
        return 'unknown'

    def parse_speaker_turns(self, text: str, section: str = '') -> List[SpeakerTurn]:
        """
        Parse text into speaker turns.

        This is a best-effort parser - transcript formats vary widely.
        """
        turns = []
        current_speaker = None
        current_role = 'unknown'
        current_text = []
        turn_index = 0

        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line starts a new speaker
            speaker_match = None
            title = ''

            for pattern in self.SPEAKER_PATTERNS:
                match = re.match(pattern, line)
                if match:
                    speaker_match = match.group(1).strip()
                    if len(match.groups()) > 1:
                        title = match.group(2) if match.group(2) else ''
                    break

            if speaker_match:
                # Save previous speaker's turn
                if current_speaker and current_text:
                    turns.append(SpeakerTurn(
                        speaker=current_speaker,
                        role=current_role,
                        text=' '.join(current_text),
                        section=section,
                        turn_index=turn_index
                    ))
                    turn_index += 1

                # Start new speaker
                current_speaker = speaker_match
                current_role = self._detect_speaker_role(speaker_match, title)

                # Get text after speaker name
                remaining = line[len(match.group(0)):].strip()
                current_text = [remaining] if remaining else []

            else:
                # Continue current speaker
                current_text.append(line)

        # Save final speaker's turn
        if current_speaker and current_text:
            turns.append(SpeakerTurn(
                speaker=current_speaker,
                role=current_role,
                text=' '.join(current_text),
                section=section,
                turn_index=turn_index
            ))

        return turns

    def _parse_period(self, period: str) -> Tuple[Optional[int], Optional[int], str]:
        """
        Parse period string into fiscal year, quarter, and label.

        Accepts formats: '2025Q3', 'Q3 2025', 'FY2025', '2025'
        """
        period = period.strip().upper()

        # Match "2025Q3" or "Q3 2025"
        match = re.match(r'(\d{4})\s*Q(\d)', period)
        if match:
            return int(match.group(1)), int(match.group(2)), f"{match.group(1)}Q{match.group(2)}"

        match = re.match(r'Q(\d)\s*(\d{4})', period)
        if match:
            return int(match.group(2)), int(match.group(1)), f"{match.group(2)}Q{match.group(1)}"

        # Match "FY2025" or just "2025"
        match = re.match(r'(?:FY)?(\d{4})', period)
        if match:
            return int(match.group(1)), None, f"FY{match.group(1)}"

        # Fallback
        return None, None, period

    def save_document(self, ticker: str, text: str, period: str,
                      sections: List[TranscriptSection] = None,
                      source: str = 'manual',
                      call_date: datetime = None,
                      metadata: Dict = None) -> Optional[int]:
        """
        Save transcript document to database.

        Args:
            ticker: Stock ticker
            text: Full transcript text
            period: Period label (e.g., '2025Q3')
            sections: Parsed sections (optional)
            source: Source identifier
            call_date: Date of earnings call
            metadata: Additional metadata

        Returns:
            doc_id if saved, None if error
        """
        # Calculate hash for deduplication
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Parse period
        fiscal_year, fiscal_quarter, period_label = self._parse_period(period)

        # Use call_date or default to now
        if call_date is None:
            call_date = datetime.utcnow()

        # Count words
        word_count = len(text.split())

        # Build metadata
        doc_metadata = metadata or {}
        if sections:
            doc_metadata['sections'] = [s.name for s in sections]
            doc_metadata['section_count'] = len(sections)

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if already exists
                    cur.execute(
                        "SELECT doc_id FROM documents WHERE raw_text_sha256 = %s",
                        (text_hash,)
                    )
                    existing = cur.fetchone()
                    if existing:
                        logger.info(f"{ticker}: Transcript already exists (doc_id={existing[0]})")
                        return existing[0]

                    # Insert document
                    cur.execute("""
                        INSERT INTO documents (
                            ticker, doc_type, source, asof_ts_utc,
                            period_label, fiscal_year, fiscal_quarter, title,
                            raw_text, raw_text_sha256, section_count, word_count, metadata
                        ) VALUES (
                            %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s, %s, %s
                        ) RETURNING doc_id
                    """, (
                        ticker.upper(),
                        'transcript',
                        source,
                        call_date,
                        period_label,
                        fiscal_year,
                        fiscal_quarter,
                        f"{ticker.upper()} Earnings Call {period_label}",
                        text,
                        text_hash,
                        len(sections) if sections else 1,
                        word_count,
                        doc_metadata
                    ))

                    doc_id = cur.fetchone()[0]
                    conn.commit()

                    logger.info(f"{ticker}: Saved transcript {period_label} (doc_id={doc_id}, {word_count:,} words)")
                    return doc_id

        except Exception as e:
            logger.error(f"Error saving transcript for {ticker}: {e}")
            return None

    def ingest_text(self, ticker: str, text: str, period: str,
                    call_date: datetime = None, metadata: Dict = None) -> Optional[int]:
        """
        Ingest transcript from raw text.

        Args:
            ticker: Stock ticker
            text: Raw transcript text
            period: Period label (e.g., '2025Q3')
            call_date: Date of earnings call
            metadata: Additional metadata

        Returns:
            doc_id if successful
        """
        # Clean text
        text = self._clean_text(text)

        if len(text) < 100:
            logger.error(f"{ticker}: Text too short ({len(text)} chars)")
            return None

        # Detect sections
        sections = self.detect_sections(text)

        # Save document
        doc_id = self.save_document(
            ticker=ticker,
            text=text,
            period=period,
            sections=sections,
            source='manual_text',
            call_date=call_date,
            metadata=metadata
        )

        return doc_id

    def ingest_file(self, ticker: str, file_path: str, period: str,
                    call_date: datetime = None, metadata: Dict = None) -> Optional[int]:
        """
        Ingest transcript from file.

        Args:
            ticker: Stock ticker
            file_path: Path to transcript file
            period: Period label (e.g., '2025Q3')
            call_date: Date of earnings call
            metadata: Additional metadata

        Returns:
            doc_id if successful
        """
        logger.info(f"{ticker}: Ingesting transcript from {file_path}")

        # Extract text from file
        try:
            text = self._extract_text_from_file(file_path)
        except Exception as e:
            logger.error(f"{ticker}: Failed to extract text from {file_path}: {e}")
            return None

        # Clean text
        text = self._clean_text(text)

        if len(text) < 100:
            logger.error(f"{ticker}: Extracted text too short ({len(text)} chars)")
            return None

        # Detect sections
        sections = self.detect_sections(text)

        # Build metadata
        file_metadata = metadata or {}
        file_metadata['source_file'] = str(file_path)
        file_metadata['file_size'] = os.path.getsize(file_path)

        # Save document
        doc_id = self.save_document(
            ticker=ticker,
            text=text,
            period=period,
            sections=sections,
            source='manual_file',
            call_date=call_date,
            metadata=file_metadata
        )

        return doc_id

    def ingest_uploaded_file(self, ticker: str, uploaded_file, period: str,
                             call_date: datetime = None) -> Optional[int]:
        """
        Ingest transcript from Streamlit uploaded file.

        Args:
            ticker: Stock ticker
            uploaded_file: Streamlit UploadedFile object
            period: Period label
            call_date: Date of earnings call

        Returns:
            doc_id if successful
        """
        file_name = uploaded_file.name
        suffix = Path(file_name).suffix.lower()

        logger.info(f"{ticker}: Processing uploaded file {file_name}")

        # Read file content
        content = uploaded_file.read()

        if suffix == '.txt':
            text = content.decode('utf-8', errors='ignore')

        elif suffix in ['.html', '.htm']:
            if not HTML_AVAILABLE:
                logger.error("beautifulsoup4/html2text required for HTML files")
                return None

            html_content = content.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html_content, 'html.parser')
            for tag in soup(['script', 'style']):
                tag.decompose()
            text = self.html_converter.handle(str(soup))

        elif suffix == '.pdf':
            if not PDF_AVAILABLE:
                logger.error("pdfplumber required for PDF files")
                return None

            # Save temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                text_parts = []
                with pdfplumber.open(tmp_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                text = '\n\n'.join(text_parts)
            finally:
                os.unlink(tmp_path)

        else:
            # Try as text
            text = content.decode('utf-8', errors='ignore')

        # Clean and process
        text = self._clean_text(text)

        if len(text) < 100:
            logger.error(f"{ticker}: Extracted text too short")
            return None

        sections = self.detect_sections(text)

        doc_id = self.save_document(
            ticker=ticker,
            text=text,
            period=period,
            sections=sections,
            source='streamlit_upload',
            call_date=call_date,
            metadata={'original_filename': file_name}
        )

        return doc_id


# Convenience function
def ingest_transcript(ticker: str, text_or_path: str, period: str,
                      call_date: datetime = None) -> Optional[int]:
    """
    Quick function to ingest a transcript.

    Automatically detects if input is a file path or raw text.
    """
    ingester = TranscriptIngester()

    if os.path.isfile(text_or_path):
        return ingester.ingest_file(ticker, text_or_path, period, call_date)
    else:
        return ingester.ingest_text(ticker, text_or_path, period, call_date)


if __name__ == "__main__":
    # Test with sample text
    sample_transcript = """
    Micron Technology Q3 2025 Earnings Call

    Operator:
    Good afternoon. My name is Sarah and I will be your conference operator today.
    Welcome to Micron Technology's Third Quarter 2025 Earnings Call.

    Prepared Remarks

    Sanjay Mehrotra - CEO:
    Thank you, Sarah. Good afternoon everyone. I'm pleased to report another strong quarter.
    Revenue was $8.7 billion, up 15% sequentially and 93% year over year.

    AI demand continues to drive unprecedented growth in our HBM portfolio.
    We shipped over 50% more HBM bits than the prior quarter.

    Mark Murphy - CFO:
    Thank you, Sanjay. Let me provide more details on our financial results.
    Gross margin improved to 35.5%, up from 31% in the prior quarter.

    Question-and-Answer Session

    Operator:
    Our first question comes from Timothy Arcuri with UBS.

    Timothy Arcuri - UBS - Analyst:
    Hi, thanks. Sanjay, can you talk about HBM capacity plans for calendar 2026?

    Sanjay Mehrotra - CEO:
    Yes, Tim. We're significantly expanding our HBM capacity.
    We expect to at least double our HBM output in 2026.
    """

    ingester = TranscriptIngester()

    # Test section detection
    print("Testing section detection:")
    sections = ingester.detect_sections(sample_transcript)
    for s in sections:
        print(f"  - {s.name}: {len(s.text)} chars")

    # Test speaker parsing
    print("\nTesting speaker detection:")
    turns = ingester.parse_speaker_turns(sample_transcript)
    for t in turns[:5]:
        print(f"  [{t.role}] {t.speaker}: {t.text[:50]}...")

    print("\n✅ Transcript ingestion module ready")