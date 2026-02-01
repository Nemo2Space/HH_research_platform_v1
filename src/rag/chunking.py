"""
Document Chunking and Embedding Pipeline
=========================================

Chunks documents and generates embeddings for RAG retrieval.

Features:
- Intelligent chunking by document type
- Token-based splitting with overlap
- Section-aware chunking for transcripts
- Batch embedding generation with bge-large-en-v1.5
- Stores in doc_chunks table with pgvector

Chunking Strategy:
- Transcripts: 900 tokens, 150 overlap, section-aware
- Filings: 1200 tokens, 200 overlap, section-aware

Usage:
    from src.rag.chunking import ChunkingPipeline

    pipeline = ChunkingPipeline()

    # Chunk and embed a single document
    chunk_ids = pipeline.process_document(doc_id)

    # Process all unprocessed documents
    pipeline.process_all_pending()

Author: HH Research Platform
"""

import os
import re
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import hashlib
from datetime import datetime, timezone
from psycopg2.extras import Json
from src.db.connection import get_connection, get_engine
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")

# Try to import tiktoken for accurate token counting
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.debug("tiktoken not installed - using approximate token counting")


@dataclass
class Chunk:
    """Represents a document chunk."""
    doc_id: int
    ticker: str
    doc_type: str
    asof_ts_utc: str
    section: Optional[str]
    speaker: Optional[str]
    speaker_role: Optional[str]
    chunk_index: int
    text: str
    approx_token_count: int
    char_start: int
    char_end: int
    metadata: Dict


@dataclass
class ChunkingConfig:
    """Configuration for chunking."""
    chunk_size: int  # Target tokens per chunk
    chunk_overlap: int  # Overlap tokens
    min_chunk_size: int  # Minimum tokens (don't create tiny chunks)

    @classmethod
    def for_transcripts(cls) -> 'ChunkingConfig':
        """Config optimized for earnings transcripts."""
        return cls(
            chunk_size=900,
            chunk_overlap=150,
            min_chunk_size=100
        )

    @classmethod
    def for_filings(cls) -> 'ChunkingConfig':
        """Config optimized for SEC filings."""
        return cls(
            chunk_size=1200,
            chunk_overlap=200,
            min_chunk_size=150
        )


class TokenCounter:
    """Counts tokens in text."""

    def __init__(self):
        if TIKTOKEN_AVAILABLE:
            # Use cl100k_base (GPT-4 tokenizer) as approximation
            self.encoder = tiktoken.get_encoding("cl100k_base")
            self.mode = 'tiktoken'
        else:
            self.encoder = None
            self.mode = 'approximate'

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if self.mode == 'tiktoken':
            return len(self.encoder.encode(text))
        else:
            # Approximate: ~4 chars per token
            return len(text) // 4


class ChunkingPipeline:
    """
    Pipeline for chunking documents and generating embeddings.
    """

    # Section patterns for transcripts
    TRANSCRIPT_SECTIONS = {
        'prepared_remarks': [
            r'(?:^|\n)\s*(?:Prepared\s+Remarks|Opening\s+Remarks|Presentation)\s*(?:\n|:)',
        ],
        'qa': [
            r'(?:^|\n)\s*(?:Question[s]?\s*(?:and|&)\s*Answer[s]?|Q\s*&\s*A)\s*(?:\n|:)',
        ],
    }

    # Section patterns for filings
    FILING_SECTIONS = {
        'risk_factors': r'(?:ITEM\s*1A[.\s]*[-–]?\s*)?RISK\s*FACTORS',
        'mda': r'(?:ITEM\s*7[.\s]*[-–]?\s*)?MANAGEMENT[\'\']?S?\s*DISCUSSION',
        'business': r'(?:ITEM\s*1[.\s]*[-–]?\s*)?BUSINESS(?!\s*RISK)',
        'legal': r'(?:ITEM\s*3[.\s]*[-–]?\s*)?LEGAL\s*PROCEEDINGS',
    }

    # Speaker patterns for transcripts
    SPEAKER_PATTERN = re.compile(
        r'^([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\s*[-–,]\s*([^:\n]+?)(?::|$)',
        re.MULTILINE
    )

    def __init__(self, embedding_model: str = 'BAAI/bge-large-en-v1.5'):
        """
        Initialize the chunking pipeline.

        Args:
            embedding_model: HuggingFace model name for embeddings
        """
        self.approx_token_counter = TokenCounter()
        self.embedding_model_name = embedding_model
        self._embedder = None  # Lazy load

        # Config per doc type
        self.configs = {
            'transcript': ChunkingConfig.for_transcripts(),
            '10k': ChunkingConfig.for_filings(),
            '10q': ChunkingConfig.for_filings(),
            '8k': ChunkingConfig.for_filings(),
            'default': ChunkingConfig.for_filings(),
        }

    @property
    def embedder(self):
        """Lazy load embedding model."""
        if self._embedder is None:
            if not EMBEDDINGS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers required. Install with: "
                    "pip install sentence-transformers"
                )
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedder = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Embedding model loaded (dim={self._embedder.get_sentence_embedding_dimension()})")
        return self._embedder

    def _get_config(self, doc_type: str) -> ChunkingConfig:
        """Get chunking config for document type."""
        return self.configs.get(doc_type, self.configs['default'])

    def _detect_sections(self, text: str, doc_type: str) -> List[Tuple[str, int, int]]:
        """
        Detect sections in document.

        Returns list of (section_name, start_idx, end_idx).
        """
        if doc_type == 'transcript':
            patterns = self.TRANSCRIPT_SECTIONS
        elif doc_type in ('10k', '10q'):
            patterns = self.FILING_SECTIONS
        else:
            return [('full_text', 0, len(text))]

        # Find all section markers
        section_positions = []
        for section_name, pattern_list in patterns.items():
            if isinstance(pattern_list, str):
                pattern_list = [pattern_list]
            for pattern in pattern_list:
                for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                    section_positions.append((section_name, match.start()))

        if not section_positions:
            return [('full_text', 0, len(text))]

        # Sort by position
        section_positions.sort(key=lambda x: x[1])

        # Build section ranges
        sections = []
        for i, (name, start) in enumerate(section_positions):
            if i + 1 < len(section_positions):
                end = section_positions[i + 1][1]
            else:
                end = len(text)
            sections.append((name, start, end))

        return sections

    def _detect_speaker(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect speaker in chunk text.

        Returns (speaker_name, speaker_role).
        """
        match = self.SPEAKER_PATTERN.search(text[:500])  # Only check beginning
        if match:
            speaker = match.group(1).strip()
            title = match.group(2).strip() if match.group(2) else ''

            # Determine role
            title_lower = title.lower()
            if any(t in title_lower for t in ['ceo', 'cfo', 'coo', 'president', 'vp', 'director']):
                role = 'executive'
            elif any(t in title_lower for t in ['analyst', 'research', 'capital', 'securities']):
                role = 'analyst'
            elif 'operator' in title_lower:
                role = 'operator'
            else:
                role = 'unknown'

            return speaker, role

        return None, None

    def _split_into_chunks(self, text: str, config: ChunkingConfig,
                           section: str = None) -> List[Dict]:
        """
        Split text into chunks with overlap.

        Returns list of chunk dicts with text and positions.
        """
        chunks = []

        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)

        current_chunk = []
        current_tokens = 0
        chunk_start = 0
        char_pos = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                char_pos += 2  # Account for removed newlines
                continue

            para_tokens = self.approx_token_counter.count(para)

            # If single paragraph exceeds chunk size, split it by sentences
            if para_tokens > config.chunk_size:
                # Save current chunk first
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'char_start': chunk_start,
                        'char_end': char_pos,
                        'approx_token_count': current_tokens,
                        'section': section,
                    })
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sent_chunk = []
                sent_tokens = 0
                sent_start = char_pos

                for sent in sentences:
                    sent_tok = self.approx_token_counter.count(sent)

                    if sent_tokens + sent_tok > config.chunk_size and sent_chunk:
                        chunk_text = ' '.join(sent_chunk)
                        chunks.append({
                            'text': chunk_text,
                            'char_start': sent_start,
                            'char_end': char_pos,
                            'approx_token_count': sent_tokens,
                            'section': section,
                        })

                        # Keep overlap
                        overlap_tokens = 0
                        overlap_sents = []
                        for s in reversed(sent_chunk):
                            s_tok = self.approx_token_counter.count(s)
                            if overlap_tokens + s_tok <= config.chunk_overlap:
                                overlap_sents.insert(0, s)
                                overlap_tokens += s_tok
                            else:
                                break

                        sent_chunk = overlap_sents
                        sent_tokens = overlap_tokens
                        sent_start = char_pos - len(' '.join(overlap_sents))

                    sent_chunk.append(sent)
                    sent_tokens += sent_tok
                    char_pos += len(sent) + 1

                if sent_chunk:
                    chunk_text = ' '.join(sent_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'char_start': sent_start,
                        'char_end': char_pos,
                        'approx_token_count': sent_tokens,
                        'section': section,
                    })

                chunk_start = char_pos
                continue

            # Check if adding paragraph exceeds chunk size
            if current_tokens + para_tokens > config.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'char_start': chunk_start,
                    'char_end': char_pos,
                    'approx_token_count': current_tokens,
                    'section': section,
                })

                # Keep overlap paragraphs
                overlap_tokens = 0
                overlap_paras = []
                for p in reversed(current_chunk):
                    p_tok = self.approx_token_counter.count(p)
                    if overlap_tokens + p_tok <= config.chunk_overlap:
                        overlap_paras.insert(0, p)
                        overlap_tokens += p_tok
                    else:
                        break

                current_chunk = overlap_paras
                current_tokens = overlap_tokens
                chunk_start = char_pos - sum(len(p) + 2 for p in overlap_paras)

            current_chunk.append(para)
            current_tokens += para_tokens
            char_pos += len(para) + 2

        # Save final chunk
        if current_chunk and current_tokens >= config.min_chunk_size:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'char_start': chunk_start,
                'char_end': char_pos,
                'approx_token_count': current_tokens,
                'section': section,
            })

        return chunks

    def chunk_document(self, doc_id: int) -> List[Chunk]:
        """
        Chunk a document from the database.

        Args:
            doc_id: Document ID

        Returns:
            List of Chunk objects
        """
        # Fetch document
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ticker, doc_type, asof_ts_utc, raw_text, metadata
                    FROM rag.documents
                    WHERE doc_id = %s
                """, (doc_id,))
                row = cur.fetchone()

                if not row:
                    logger.error(f"Document {doc_id} not found")
                    return []

                ticker, doc_type, asof_ts_utc, raw_text, metadata = row

        config = self._get_config(doc_type)

        # Detect sections
        sections = self._detect_sections(raw_text, doc_type)

        all_chunks = []
        chunk_index = 0

        for section_name, start, end in sections:
            section_text = raw_text[start:end]

            # Split section into chunks
            chunk_dicts = self._split_into_chunks(section_text, config, section_name)

            for cd in chunk_dicts:
                # Adjust positions to document-level
                cd['char_start'] += start
                cd['char_end'] += start

                # Detect speaker for transcripts
                speaker, speaker_role = None, None
                if doc_type == 'transcript':
                    speaker, speaker_role = self._detect_speaker(cd['text'])

                chunk = Chunk(
                    doc_id=doc_id,
                    ticker=ticker,
                    doc_type=doc_type,
                    asof_ts_utc=str(asof_ts_utc),
                    section=cd['section'],
                    speaker=speaker,
                    speaker_role=speaker_role,
                    chunk_index=chunk_index,
                    text=cd['text'],
                    approx_token_count=cd['approx_token_count'],
                    char_start=cd['char_start'],
                    char_end=cd['char_end'],
                    metadata={}
                )
                all_chunks.append(chunk)
                chunk_index += 1

        logger.info(f"Document {doc_id} ({ticker} {doc_type}): Created {len(all_chunks)} chunks")
        return all_chunks

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for embedding

        Returns:
            numpy array of embeddings
        """
        embeddings = self.embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True  # For cosine similarity
        )
        return embeddings

    def save_chunks(self, chunks: List[Chunk], embeddings: np.ndarray) -> List[int]:
        """
        Save chunks with embeddings to database.

        Returns list of chunk_ids.
        """
        chunk_ids = []

        with get_connection() as conn:
            with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings):
                    # Convert numpy array to list for psycopg2
                    embedding_list = embedding.tolist()

                    cur.execute("""
                        INSERT INTO rag.doc_chunks (
                            doc_id, ticker, doc_type, asof_ts_utc,
                            section, speaker, speaker_role, chunk_index,
                            text, approx_token_count, char_len, char_start, char_end,
                            chunk_hash, embedding, embedding_model, embedding_dim, embedded_at_utc, metadata
                        ) VALUES (
                            %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s
                        ) RETURNING chunk_id
                    """, (
                        chunk.doc_id, chunk.ticker, chunk.doc_type, chunk.asof_ts_utc,
                        chunk.section, chunk.speaker, chunk.speaker_role, chunk.chunk_index,
                        chunk.text, chunk.approx_token_count, len(chunk.text), chunk.char_start, chunk.char_end,
                        hashlib.sha256(chunk.text.encode()).hexdigest(), embedding_list, self.embedding_model_name, 1024, datetime.now(timezone.utc), Json(chunk.metadata)
                    ))

                    chunk_id = cur.fetchone()[0]
                    chunk_ids.append(chunk_id)

                conn.commit()

        logger.info(f"Saved {len(chunk_ids)} chunks to database")
        return chunk_ids

    def process_document(self, doc_id: int) -> List[int]:
        """
        Full pipeline: chunk document and generate embeddings.

        Args:
            doc_id: Document ID to process

        Returns:
            List of chunk_ids created
        """
        # Check if already processed
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM rag.doc_chunks WHERE doc_id = %s",
                    (doc_id,)
                )
                existing = cur.fetchone()[0]
                if existing > 0:
                    logger.info(f"Document {doc_id} already has {existing} chunks - skipping")
                    return []

        # Chunk document
        chunks = self.chunk_document(doc_id)
        if not chunks:
            return []

        # Generate embeddings
        texts = [c.text for c in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.generate_embeddings(texts)

        # Save to database
        chunk_ids = self.save_chunks(chunks, embeddings)

        return chunk_ids

    def process_all_pending(self, limit: int = None) -> Dict:
        """
        Process all documents that haven't been chunked yet.

        Args:
            limit: Maximum documents to process (None = all)

        Returns:
            Summary dict
        """
        # Find documents without chunks
        with get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT d.doc_id, d.ticker, d.doc_type
                    FROM rag.documents d
                    LEFT JOIN rag.doc_chunks c ON d.doc_id = c.doc_id
                    WHERE c.chunk_id IS NULL
                    ORDER BY d.ingested_at_utc DESC
                """
                if limit:
                    query += f" LIMIT {limit}"

                cur.execute(query)
                pending = cur.fetchall()

        if not pending:
            logger.info("No pending documents to process")
            return {'processed': 0, 'total_chunks': 0}

        logger.info(f"Processing {len(pending)} pending documents")

        results = {
            'processed': 0,
            'failed': 0,
            'total_chunks': 0,
            'by_ticker': {}
        }

        for doc_id, ticker, doc_type in pending:
            try:
                chunk_ids = self.process_document(doc_id)
                results['processed'] += 1
                results['total_chunks'] += len(chunk_ids)

                if ticker not in results['by_ticker']:
                    results['by_ticker'][ticker] = {'docs': 0, 'chunks': 0}
                results['by_ticker'][ticker]['docs'] += 1
                results['by_ticker'][ticker]['chunks'] += len(chunk_ids)

            except Exception as e:
                logger.error(f"Failed to process doc {doc_id} ({ticker}): {e}")
                results['failed'] += 1

        logger.info(f"Processing complete: {results['processed']} docs, "
                    f"{results['total_chunks']} chunks, {results['failed']} failed")

        return results

    def reprocess_document(self, doc_id: int) -> List[int]:
        """
        Delete existing chunks and reprocess document.

        Args:
            doc_id: Document ID

        Returns:
            List of new chunk_ids
        """
        # Delete existing chunks
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM rag.doc_chunks WHERE doc_id = %s", (doc_id,))
                deleted = cur.rowcount
                conn.commit()

        if deleted > 0:
            logger.info(f"Deleted {deleted} existing chunks for doc {doc_id}")

        # Process fresh
        return self.process_document(doc_id)


# Convenience functions
def chunk_and_embed_document(doc_id: int) -> List[int]:
    """Quick function to process a single document."""
    pipeline = ChunkingPipeline()
    return pipeline.process_document(doc_id)


def process_pending_documents(limit: int = None) -> Dict:
    """Quick function to process all pending documents."""
    pipeline = ChunkingPipeline()
    return pipeline.process_all_pending(limit)


if __name__ == "__main__":
    import sys

    pipeline = ChunkingPipeline()

    if len(sys.argv) > 1:
        # Process specific document
        doc_id = int(sys.argv[1])
        print(f"Processing document {doc_id}...")
        chunk_ids = pipeline.process_document(doc_id)
        print(f"Created {len(chunk_ids)} chunks: {chunk_ids[:5]}...")
    else:
        # Process all pending
        print("Processing all pending documents...")
        results = pipeline.process_all_pending()
        print(f"Results: {results}")