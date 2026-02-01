"""
RAG Structured Fact Extractor (Batch C) - Fixed for Exact Schema Types
======================================================================

Extracts structured facts from documents using LLM.
Properly handles JSONB vs TEXT vs ARRAY column types.

Author: HH Research Platform
"""

import os
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from psycopg2.extras import Json
from src.db.connection import get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


RISK_THEMES_TAXONOMY = [
    'supply_chain', 'china_exposure', 'pricing_pressure', 'demand_weakness',
    'regulation', 'competition', 'currency', 'capex_overrun', 'margin_compression',
    'customer_concentration', 'inventory', 'labor', 'cybersecurity', 'litigation',
    'interest_rates', 'inflation', 'geopolitical', 'environmental', 'product_quality',
    'key_customer_loss',
]


@dataclass
class FilingFact:
    """Extracted facts from SEC filings - exact schema match."""
    doc_id: int
    ticker: str
    filing_type: str
    period_label: str

    # ARRAY columns
    key_risks: List[str] = field(default_factory=list)
    new_risks: List[str] = field(default_factory=list)
    removed_risks: List[str] = field(default_factory=list)

    # JSONB columns
    risk_severity: Dict[str, Any] = field(default_factory=dict)
    segment_notes: Dict[str, Any] = field(default_factory=dict)
    geographic_risks: Dict[str, Any] = field(default_factory=dict)
    capex_plan: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    raw_json: Dict[str, Any] = field(default_factory=dict)

    # TEXT columns
    litigation_notes: Optional[str] = None
    cybersecurity_notes: Optional[str] = None
    china_exposure: Optional[str] = None

    # BOOLEAN columns
    material_litigation: bool = False
    cyber_incidents: bool = False

    # Metadata
    extractor_model: str = ""
    extractor_version: str = "1.0"


@dataclass
class TranscriptFact:
    """Extracted facts from transcripts - exact schema match."""
    doc_id: int
    ticker: str
    period_label: str

    # TEXT columns
    guidance_direction: Optional[str] = None
    guidance_notes: Optional[str] = None
    demand_tone: Optional[str] = None
    ai_sentiment: Optional[str] = None
    ai_commentary: Optional[str] = None
    capex_direction: Optional[str] = None
    capex_notes: Optional[str] = None
    margin_outlook: Optional[str] = None
    margin_notes: Optional[str] = None

    # NUMERIC columns
    guidance_revenue_low: Optional[float] = None
    guidance_revenue_high: Optional[float] = None
    guidance_eps_low: Optional[float] = None
    guidance_eps_high: Optional[float] = None
    capex_amount: Optional[float] = None
    ai_mentions_count: int = 0
    extraction_confidence: float = 0.5

    # ARRAY columns
    demand_drivers: List[str] = field(default_factory=list)
    risk_themes: List[str] = field(default_factory=list)
    risk_phrases_raw: List[str] = field(default_factory=list)

    # JSONB columns
    evidence: Dict[str, Any] = field(default_factory=dict)
    raw_json: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    extractor_model: str = ""
    extractor_version: str = "1.0"


@dataclass
class ExtractorConfig:
    llm_base_url: str = "http://localhost:8090/v1"
    llm_model: str = "qwen3-32b"
    llm_api_key: str = "not-needed"
    llm_max_tokens: int = 2000
    llm_temperature: float = 0.1
    fallback_enabled: bool = True
    fallback_base_url: str = "https://api.openai.com/v1"
    fallback_model: str = "gpt-4o-mini"
    fallback_api_key: Optional[str] = None
    max_chunks_per_extraction: int = 10

    @classmethod
    def from_env(cls) -> 'ExtractorConfig':
        return cls(
            llm_base_url=os.getenv('LLM_BASE_URL', 'http://localhost:8090/v1'),
            llm_model=os.getenv('LLM_MODEL', 'qwen3-32b'),
            fallback_api_key=os.getenv('OPENAI_API_KEY'),
        )


FILING_EXTRACTION_PROMPT = """You are a financial analyst extracting structured facts from an SEC filing.

FILING CONTENT:
{content}

Extract information in this exact JSON format:

{{
    "key_risks": ["risk1", "risk2"],
    "risk_severity": {{"overall": "high", "details": "explanation"}},
    "new_risks": ["new risk descriptions"],
    "removed_risks": [],
    "segment_notes": {{"summary": "segment performance notes"}},
    "litigation_notes": "material litigation summary or null",
    "material_litigation": false,
    "cybersecurity_notes": "cyber disclosure or null",
    "cyber_incidents": false,
    "china_exposure": "China/Taiwan exposure description or null",
    "geographic_risks": {{"regions": ["Asia", "Europe"], "details": "specifics"}},
    "capex_plan": {{"direction": "increasing", "amount": null, "notes": "details"}}
}}

Valid risk themes: {risk_themes}

Return ONLY valid JSON."""


TRANSCRIPT_EXTRACTION_PROMPT = """You are a financial analyst extracting facts from an earnings call transcript.

TRANSCRIPT CONTENT:
{content}

Extract information in this exact JSON format:

{{
    "guidance_direction": "raised" | "maintained" | "lowered" | null,
    "guidance_revenue_low": null,
    "guidance_revenue_high": null,
    "guidance_eps_low": null,
    "guidance_eps_high": null,
    "guidance_notes": "guidance summary",
    "demand_tone": "bullish" | "neutral" | "cautious" | "bearish",
    "demand_drivers": ["driver1", "driver2"],
    "ai_mentions_count": 5,
    "ai_sentiment": "positive" | "neutral" | "negative" | null,
    "ai_commentary": "key AI quote",
    "capex_direction": "increasing" | "stable" | "decreasing" | null,
    "capex_amount": null,
    "capex_notes": "capex summary",
    "margin_outlook": "expanding" | "stable" | "contracting" | null,
    "margin_notes": "margin commentary",
    "risk_themes": ["theme1", "theme2"],
    "risk_phrases_raw": ["exact risk phrases"],
    "extraction_confidence": 0.8
}}

Valid risk themes: {risk_themes}

Return ONLY valid JSON."""


class FactExtractor:
    def __init__(self, config: ExtractorConfig = None):
        self.config = config or ExtractorConfig.from_env()
        self._llm_client = None
        self._fallback_client = None

    def _get_llm_client(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required")
        if self._llm_client is None:
            self._llm_client = OpenAI(
                base_url=self.config.llm_base_url,
                api_key=self.config.llm_api_key,
            )
        return self._llm_client

    def _get_fallback_client(self):
        if not self.config.fallback_enabled or not self.config.fallback_api_key:
            return None
        if self._fallback_client is None:
            self._fallback_client = OpenAI(
                base_url=self.config.fallback_base_url,
                api_key=self.config.fallback_api_key,
            )
        return self._fallback_client

    def _call_llm(self, prompt: str, use_fallback: bool = False) -> str:
        client = self._get_fallback_client() if use_fallback else self._get_llm_client()
        model = self.config.fallback_model if use_fallback else self.config.llm_model

        if client is None:
            raise RuntimeError("No LLM client available")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            if not use_fallback and self._get_fallback_client():
                logger.warning(f"Primary LLM failed, trying fallback: {e}")
                return self._call_llm(prompt, use_fallback=True)
            raise

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {}

    def _get_document_chunks(self, doc_id: int, limit: int = 10) -> Tuple[str, List[int]]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT chunk_id, text 
                    FROM rag.doc_chunks 
                    WHERE doc_id = %s
                    ORDER BY chunk_index
                    LIMIT %s
                """, (doc_id, limit))
                rows = cur.fetchall()

        chunk_ids = [row[0] for row in rows]
        content = "\n\n---\n\n".join([row[1] for row in rows])
        return content, chunk_ids

    def _get_document_info(self, doc_id: int) -> Dict[str, Any]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ticker, doc_type, period_label
                    FROM rag.documents
                    WHERE doc_id = %s
                """, (doc_id,))
                row = cur.fetchone()

        if not row:
            raise ValueError(f"Document {doc_id} not found")

        return {'ticker': row[0], 'doc_type': row[1], 'period_label': row[2]}

    def extract_filing_facts(self, doc_id: int) -> Optional[FilingFact]:
        logger.info(f"Extracting filing facts for doc_id={doc_id}")

        doc_info = self._get_document_info(doc_id)
        content, chunk_ids = self._get_document_chunks(doc_id, self.config.max_chunks_per_extraction)

        if not content:
            logger.error(f"No chunks found for doc_id={doc_id}")
            return None

        prompt = FILING_EXTRACTION_PROMPT.format(
            content=content[:15000],
            risk_themes=", ".join(RISK_THEMES_TAXONOMY),
        )

        try:
            response = self._call_llm(prompt)
            data = self._parse_json_response(response)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return None

        if not data:
            return None

        # Build fact with proper types
        fact = FilingFact(
            doc_id=doc_id,
            ticker=doc_info['ticker'],
            filing_type=doc_info['doc_type'],
            period_label=doc_info['period_label'] or '',
            # ARRAY columns - ensure lists
            key_risks=data.get('key_risks', []) or [],
            new_risks=data.get('new_risks', []) or [],
            removed_risks=data.get('removed_risks', []) or [],
            # JSONB columns - ensure dicts
            risk_severity=data.get('risk_severity') if isinstance(data.get('risk_severity'), dict) else {'level': data.get('risk_severity')},
            segment_notes=data.get('segment_notes') if isinstance(data.get('segment_notes'), dict) else {'notes': data.get('segment_notes')},
            geographic_risks=data.get('geographic_risks') if isinstance(data.get('geographic_risks'), dict) else {'regions': data.get('geographic_risks', [])},
            capex_plan=data.get('capex_plan') if isinstance(data.get('capex_plan'), dict) else {'notes': data.get('capex_plan')},
            evidence={'chunk_ids': chunk_ids},
            raw_json=data,
            # TEXT columns
            litigation_notes=data.get('litigation_notes'),
            cybersecurity_notes=data.get('cybersecurity_notes'),
            china_exposure=data.get('china_exposure'),
            # BOOLEAN columns
            material_litigation=bool(data.get('material_litigation', False)),
            cyber_incidents=bool(data.get('cyber_incidents', False)),
            # Metadata
            extractor_model=self.config.fallback_model,
            extractor_version="1.0",
        )

        return fact

    def extract_transcript_facts(self, doc_id: int) -> Optional[TranscriptFact]:
        logger.info(f"Extracting transcript facts for doc_id={doc_id}")

        doc_info = self._get_document_info(doc_id)
        content, chunk_ids = self._get_document_chunks(doc_id, self.config.max_chunks_per_extraction)

        if not content:
            return None

        prompt = TRANSCRIPT_EXTRACTION_PROMPT.format(
            content=content[:15000],
            risk_themes=", ".join(RISK_THEMES_TAXONOMY),
        )

        try:
            response = self._call_llm(prompt)
            data = self._parse_json_response(response)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return None

        if not data:
            return None

        fact = TranscriptFact(
            doc_id=doc_id,
            ticker=doc_info['ticker'],
            period_label=doc_info['period_label'] or '',
            guidance_direction=data.get('guidance_direction'),
            guidance_revenue_low=data.get('guidance_revenue_low'),
            guidance_revenue_high=data.get('guidance_revenue_high'),
            guidance_eps_low=data.get('guidance_eps_low'),
            guidance_eps_high=data.get('guidance_eps_high'),
            guidance_notes=data.get('guidance_notes'),
            demand_tone=data.get('demand_tone'),
            demand_drivers=data.get('demand_drivers', []) or [],
            ai_mentions_count=data.get('ai_mentions_count', 0) or 0,
            ai_sentiment=data.get('ai_sentiment'),
            ai_commentary=data.get('ai_commentary'),
            capex_direction=data.get('capex_direction'),
            capex_amount=data.get('capex_amount'),
            capex_notes=data.get('capex_notes'),
            margin_outlook=data.get('margin_outlook'),
            margin_notes=data.get('margin_notes'),
            risk_themes=data.get('risk_themes', []) or [],
            risk_phrases_raw=data.get('risk_phrases_raw', []) or [],
            evidence={'chunk_ids': chunk_ids},
            raw_json=data,
            extraction_confidence=data.get('extraction_confidence', 0.5) or 0.5,
            extractor_model=self.config.fallback_model,
        )

        return fact

    def save_filing_fact(self, fact: FilingFact) -> int:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Get asof_ts_utc from documents table
                cur.execute("SELECT asof_ts_utc FROM rag.documents WHERE doc_id = %s", (fact.doc_id,))
                row = cur.fetchone()
                asof_ts_utc = row[0] if row else datetime.now(timezone.utc)

                cur.execute("""
                    INSERT INTO rag.filing_facts (
                        doc_id, ticker, filing_type, asof_ts_utc, period_label,
                        key_risks, risk_severity, new_risks, removed_risks,
                        segment_notes, litigation_notes, material_litigation,
                        cybersecurity_notes, cyber_incidents,
                        china_exposure, geographic_risks, capex_plan,
                        evidence, raw_json,
                        extractor_model, extractor_version
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s
                    )
                    ON CONFLICT (doc_id) DO UPDATE SET
                        key_risks = EXCLUDED.key_risks,
                        risk_severity = EXCLUDED.risk_severity,
                        raw_json = EXCLUDED.raw_json,
                        extracted_at_utc = NOW()
                    RETURNING fact_id
                """, (
                    fact.doc_id, fact.ticker, fact.filing_type, asof_ts_utc, fact.period_label,
                    fact.key_risks,
                    Json(fact.risk_severity),
                    fact.new_risks,
                    fact.removed_risks,
                    Json(fact.segment_notes),
                    fact.litigation_notes,
                    fact.material_litigation,
                    fact.cybersecurity_notes,
                    fact.cyber_incidents,
                    fact.china_exposure,
                    Json(fact.geographic_risks),
                    Json(fact.capex_plan),
                    Json(fact.evidence),
                    Json(fact.raw_json),
                    fact.extractor_model,
                    fact.extractor_version,
                ))
                fact_id = cur.fetchone()[0]
                conn.commit()

        logger.info(f"Saved filing fact {fact_id} for {fact.ticker}")
        return fact_id

    def save_transcript_fact(self, fact: TranscriptFact) -> int:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO rag.transcript_facts (
                        doc_id, ticker, period_label,
                        guidance_direction, guidance_revenue_low, guidance_revenue_high,
                        guidance_eps_low, guidance_eps_high, guidance_notes,
                        demand_tone, demand_drivers,
                        ai_mentions_count, ai_sentiment, ai_commentary,
                        capex_direction, capex_amount, capex_notes,
                        margin_outlook, margin_notes,
                        risk_themes, risk_phrases_raw,
                        evidence, raw_json,
                        extractor_model, extractor_version, extraction_confidence
                    ) VALUES (
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s, %s,
                        %s, %s,
                        %s, %s, %s
                    )
                    ON CONFLICT (doc_id) DO UPDATE SET
                        guidance_direction = EXCLUDED.guidance_direction,
                        demand_tone = EXCLUDED.demand_tone,
                        ai_mentions_count = EXCLUDED.ai_mentions_count,
                        risk_themes = EXCLUDED.risk_themes,
                        raw_json = EXCLUDED.raw_json,
                        extracted_at_utc = NOW()
                    RETURNING fact_id
                """, (
                    fact.doc_id, fact.ticker, fact.period_label,
                    fact.guidance_direction, fact.guidance_revenue_low, fact.guidance_revenue_high,
                    fact.guidance_eps_low, fact.guidance_eps_high, fact.guidance_notes,
                    fact.demand_tone, fact.demand_drivers,
                    fact.ai_mentions_count, fact.ai_sentiment, fact.ai_commentary,
                    fact.capex_direction, fact.capex_amount, fact.capex_notes,
                    fact.margin_outlook, fact.margin_notes,
                    fact.risk_themes, fact.risk_phrases_raw,
                    Json(fact.evidence), Json(fact.raw_json),
                    fact.extractor_model, fact.extractor_version, fact.extraction_confidence,
                ))
                fact_id = cur.fetchone()[0]
                conn.commit()

        logger.info(f"Saved transcript fact {fact_id} for {fact.ticker}")
        return fact_id

    def process_document(self, doc_id: int) -> Optional[int]:
        doc_info = self._get_document_info(doc_id)
        doc_type = doc_info['doc_type']

        if doc_type == 'transcript':
            fact = self.extract_transcript_facts(doc_id)
            if fact:
                return self.save_transcript_fact(fact)
        elif doc_type in ('10k', '10q', '8k'):
            fact = self.extract_filing_facts(doc_id)
            if fact:
                return self.save_filing_fact(fact)
        else:
            logger.warning(f"Unknown document type: {doc_type}")

        return None

    def process_all_pending(self) -> Dict[str, int]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT d.doc_id, d.ticker, d.doc_type
                    FROM rag.documents d
                    LEFT JOIN rag.transcript_facts tf ON d.doc_id = tf.doc_id
                    LEFT JOIN rag.filing_facts ff ON d.doc_id = ff.doc_id
                    WHERE tf.fact_id IS NULL AND ff.fact_id IS NULL
                    ORDER BY d.ingested_at_utc DESC
                """)
                pending = cur.fetchall()

        logger.info(f"Processing {len(pending)} pending documents")

        results = {'processed': 0, 'failed': 0, 'skipped': 0}

        for doc_id, ticker, doc_type in pending:
            try:
                fact_id = self.process_document(doc_id)
                if fact_id:
                    results['processed'] += 1
                    print(f"  ✅ {ticker} ({doc_type}) -> fact_id={fact_id}")
                else:
                    results['skipped'] += 1
                    print(f"  ⚠️ {ticker} ({doc_type}) -> skipped")
            except Exception as e:
                logger.error(f"Failed to process doc {doc_id} ({ticker}): {e}")
                results['failed'] += 1
                print(f"  ❌ {ticker} ({doc_type}) -> error: {e}")

        return results


if __name__ == "__main__":
    print("Testing Fact Extractor...")

    extractor = FactExtractor()
    print(f"Config: {extractor.config.llm_model}")
    print(f"Risk themes: {len(RISK_THEMES_TAXONOMY)} categories")

    print("\nProcessing all pending documents...")
    results = extractor.process_all_pending()

    print(f"\nResults:")
    print(f"  Processed: {results['processed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Skipped: {results['skipped']}")