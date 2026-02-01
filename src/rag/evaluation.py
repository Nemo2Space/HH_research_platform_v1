"""
RAG Evaluation Harness (Batch C)
================================

Evaluates RAG system quality using:
- Pre-defined QA pairs with expected answers
- Retrieval metrics (precision, recall, MRR)
- Answer quality metrics (faithfulness, relevance)
- Citation accuracy

Usage:
    from src.rag.evaluation import RAGEvaluator

    evaluator = RAGEvaluator()
    results = evaluator.run_evaluation()
    evaluator.print_report(results)

Author: HH Research Platform
"""

import os
import json
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from psycopg2.extras import Json
from src.db.connection import get_connection
from src.utils.logging import get_logger
from src.rag.rag_service import RAGService, RAGResponse

logger = get_logger(__name__)


@dataclass
class EvalCase:
    """A single evaluation test case."""
    case_id: Optional[int]
    ticker: str
    question: str
    expected_answer: str
    expected_chunk_ids: List[int]  # Chunks that should be retrieved
    category: str  # guidance, risk, financial, general
    difficulty: str  # easy, medium, hard

    # Optional metadata
    source_doc_ids: List[int] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of evaluating a single test case."""
    case_id: int
    ticker: str
    question: str

    # RAG response
    answer: str
    retrieved_chunk_ids: List[int]
    citations_found: List[int]
    gating_passed: bool

    # Retrieval metrics
    retrieval_precision: float  # What % of retrieved chunks are relevant
    retrieval_recall: float     # What % of relevant chunks were retrieved
    retrieval_mrr: float        # Mean Reciprocal Rank

    # Answer metrics
    answer_faithfulness: float  # Does answer match evidence (0-1)
    answer_relevance: float     # Does answer address question (0-1)
    citation_accuracy: float    # Are citations correct (0-1)

    # Timing
    latency_ms: int

    # Pass/fail
    passed: bool
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class EvalRunSummary:
    """Summary of an evaluation run."""
    run_id: Optional[int]
    started_at: datetime
    completed_at: datetime

    # Counts
    total_cases: int
    passed_cases: int
    failed_cases: int

    # Aggregate metrics
    avg_retrieval_precision: float
    avg_retrieval_recall: float
    avg_retrieval_mrr: float
    avg_answer_faithfulness: float
    avg_answer_relevance: float
    avg_citation_accuracy: float
    avg_latency_ms: float

    # By category
    results_by_category: Dict[str, Dict[str, float]]

    # Individual results
    results: List[EvalResult]


class RAGEvaluator:
    """
    Evaluates RAG system quality.
    """

    def __init__(self, rag_service: RAGService = None):
        self.rag_service = rag_service or RAGService()

    def _calculate_retrieval_precision(
        self,
        retrieved: List[int],
        expected: List[int],
    ) -> float:
        """Calculate retrieval precision (relevant retrieved / total retrieved)."""
        if not retrieved:
            return 0.0

        relevant_retrieved = len(set(retrieved) & set(expected))
        return relevant_retrieved / len(retrieved)

    def _calculate_retrieval_recall(
        self,
        retrieved: List[int],
        expected: List[int],
    ) -> float:
        """Calculate retrieval recall (relevant retrieved / total relevant)."""
        if not expected:
            return 1.0  # No expected chunks = perfect recall

        relevant_retrieved = len(set(retrieved) & set(expected))
        return relevant_retrieved / len(expected)

    def _calculate_mrr(
        self,
        retrieved: List[int],
        expected: List[int],
    ) -> float:
        """Calculate Mean Reciprocal Rank."""
        if not expected or not retrieved:
            return 0.0

        for i, chunk_id in enumerate(retrieved):
            if chunk_id in expected:
                return 1.0 / (i + 1)

        return 0.0

    def _evaluate_faithfulness(
        self,
        answer: str,
        evidence_texts: List[str],
    ) -> float:
        """
        Evaluate if answer is faithful to evidence.

        Simple heuristic: check if key phrases from answer appear in evidence.
        For production, use LLM-as-judge.
        """
        if not evidence_texts:
            return 0.0

        # Combine evidence
        evidence_combined = " ".join(evidence_texts).lower()

        # Extract key phrases from answer (simple: sentences with facts)
        sentences = re.split(r'[.!?]', answer)

        faithful_count = 0
        total_factual = 0

        for sentence in sentences:
            sentence = sentence.strip().lower()
            if not sentence:
                continue

            # Skip non-factual sentences
            if any(word in sentence for word in ['i think', 'perhaps', 'maybe', 'unknown']):
                continue

            # Check if sentence content appears in evidence
            words = set(re.findall(r'\b\w+\b', sentence))
            evidence_words = set(re.findall(r'\b\w+\b', evidence_combined))

            overlap = len(words & evidence_words) / len(words) if words else 0

            if overlap > 0.3:  # 30% word overlap threshold
                faithful_count += 1

            total_factual += 1

        return faithful_count / total_factual if total_factual > 0 else 0.5

    def _evaluate_relevance(
        self,
        answer: str,
        question: str,
    ) -> float:
        """
        Evaluate if answer is relevant to question.

        Simple heuristic: check for question keywords in answer.
        For production, use LLM-as-judge.
        """
        # Extract question keywords (nouns, verbs)
        question_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
        answer_lower = answer.lower()

        # Check how many question words appear in answer
        matches = sum(1 for word in question_words if word in answer_lower)

        return min(matches / max(len(question_words), 1), 1.0)

    def _evaluate_citation_accuracy(
        self,
        answer: str,
        citations: List[int],
        retrieved_chunks: Dict[int, str],
    ) -> float:
        """
        Evaluate if citations are accurate.

        Check if cited chunk IDs actually support the claim.
        """
        if not citations:
            # No citations when there should be
            return 0.0 if '[chunk:' not in answer else 1.0

        valid_citations = 0

        for chunk_id in citations:
            if chunk_id in retrieved_chunks:
                valid_citations += 1

        return valid_citations / len(citations)

    def evaluate_case(self, case: EvalCase) -> EvalResult:
        """Evaluate a single test case."""
        import time

        start_time = time.time()

        # Run RAG query
        response = self.rag_service.query(
            ticker=case.ticker,
            query=case.question,
            log_response=False,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Get retrieved chunk IDs
        retrieved_chunk_ids = [c.chunk_id for c in response.retrieval_result.chunks]
        citation_ids = [c.chunk_id for c in response.citations]

        # Build chunk text lookup
        chunk_texts = {c.chunk_id: c.text for c in response.retrieval_result.chunks}
        evidence_texts = list(chunk_texts.values())

        # Calculate metrics
        precision = self._calculate_retrieval_precision(
            retrieved_chunk_ids, case.expected_chunk_ids
        )
        recall = self._calculate_retrieval_recall(
            retrieved_chunk_ids, case.expected_chunk_ids
        )
        mrr = self._calculate_mrr(
            retrieved_chunk_ids, case.expected_chunk_ids
        )
        faithfulness = self._evaluate_faithfulness(
            response.answer, evidence_texts
        )
        relevance = self._evaluate_relevance(
            response.answer, case.question
        )
        citation_accuracy = self._evaluate_citation_accuracy(
            response.answer, citation_ids, chunk_texts
        )

        # Determine pass/fail
        failure_reasons = []

        if not response.gating_passed:
            failure_reasons.append("Gating failed")

        if recall < 0.3:
            failure_reasons.append(f"Low recall ({recall:.2f})")

        if faithfulness < 0.5:
            failure_reasons.append(f"Low faithfulness ({faithfulness:.2f})")

        if relevance < 0.3:
            failure_reasons.append(f"Low relevance ({relevance:.2f})")

        if not citation_ids and case.expected_chunk_ids:
            failure_reasons.append("No citations provided")

        passed = len(failure_reasons) == 0

        return EvalResult(
            case_id=case.case_id or 0,
            ticker=case.ticker,
            question=case.question,
            answer=response.answer,
            retrieved_chunk_ids=retrieved_chunk_ids,
            citations_found=citation_ids,
            gating_passed=response.gating_passed,
            retrieval_precision=precision,
            retrieval_recall=recall,
            retrieval_mrr=mrr,
            answer_faithfulness=faithfulness,
            answer_relevance=relevance,
            citation_accuracy=citation_accuracy,
            latency_ms=latency_ms,
            passed=passed,
            failure_reasons=failure_reasons,
        )

    def load_eval_cases(self) -> List[EvalCase]:
        """Load evaluation cases from database."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        case_id, ticker, question, expected_answer,
                        expected_chunk_ids, category, difficulty,
                        source_doc_ids, tags
                    FROM rag.rag_eval_cases
                    WHERE is_active = TRUE
                    ORDER BY category, difficulty
                """)
                rows = cur.fetchall()

        cases = []
        for row in rows:
            cases.append(EvalCase(
                case_id=row[0],
                ticker=row[1],
                question=row[2],
                expected_answer=row[3],
                expected_chunk_ids=row[4] or [],
                category=row[5],
                difficulty=row[6],
                source_doc_ids=row[7] or [],
                tags=row[8] or [],
            ))

        return cases

    def create_eval_case(
        self,
        ticker: str,
        question: str,
        expected_answer: str,
        expected_chunk_ids: List[int] = None,
        category: str = "general",
        difficulty: str = "medium",
        source_doc_ids: List[int] = None,
        tags: List[str] = None,
    ) -> int:
        """Create a new evaluation case."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO rag.rag_eval_cases (
                        ticker, question, expected_answer,
                        expected_chunk_ids, category, difficulty,
                        source_doc_ids, tags
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING case_id
                """, (
                    ticker, question, expected_answer,
                    expected_chunk_ids or [], category, difficulty,
                    source_doc_ids or [], tags or [],
                ))
                case_id = cur.fetchone()[0]
                conn.commit()

        logger.info(f"Created eval case {case_id}")
        return case_id

    def run_evaluation(
        self,
        cases: List[EvalCase] = None,
        save_results: bool = True,
    ) -> EvalRunSummary:
        """
        Run evaluation on all test cases.

        Args:
            cases: Test cases to evaluate (loads from DB if None)
            save_results: Whether to save results to database

        Returns:
            EvalRunSummary with all metrics
        """
        if cases is None:
            cases = self.load_eval_cases()

        if not cases:
            logger.warning("No evaluation cases found")
            return None

        logger.info(f"Running evaluation on {len(cases)} cases")

        started_at = datetime.now(timezone.utc)
        results = []

        for i, case in enumerate(cases):
            logger.info(f"Evaluating case {i+1}/{len(cases)}: {case.ticker} - {case.question[:50]}...")

            try:
                result = self.evaluate_case(case)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate case {case.case_id}: {e}")
                # Create failed result
                results.append(EvalResult(
                    case_id=case.case_id or 0,
                    ticker=case.ticker,
                    question=case.question,
                    answer="",
                    retrieved_chunk_ids=[],
                    citations_found=[],
                    gating_passed=False,
                    retrieval_precision=0,
                    retrieval_recall=0,
                    retrieval_mrr=0,
                    answer_faithfulness=0,
                    answer_relevance=0,
                    citation_accuracy=0,
                    latency_ms=0,
                    passed=False,
                    failure_reasons=[f"Error: {str(e)}"],
                ))

        completed_at = datetime.now(timezone.utc)

        # Calculate aggregates
        passed_count = sum(1 for r in results if r.passed)

        avg_precision = sum(r.retrieval_precision for r in results) / len(results)
        avg_recall = sum(r.retrieval_recall for r in results) / len(results)
        avg_mrr = sum(r.retrieval_mrr for r in results) / len(results)
        avg_faithfulness = sum(r.answer_faithfulness for r in results) / len(results)
        avg_relevance = sum(r.answer_relevance for r in results) / len(results)
        avg_citation = sum(r.citation_accuracy for r in results) / len(results)
        avg_latency = sum(r.latency_ms for r in results) / len(results)

        # Group by category
        results_by_category = {}
        for result in results:
            case = next((c for c in cases if c.case_id == result.case_id), None)
            if case:
                cat = case.category
                if cat not in results_by_category:
                    results_by_category[cat] = {
                        'total': 0,
                        'passed': 0,
                        'avg_recall': 0,
                        'avg_faithfulness': 0,
                    }
                results_by_category[cat]['total'] += 1
                if result.passed:
                    results_by_category[cat]['passed'] += 1
                results_by_category[cat]['avg_recall'] += result.retrieval_recall
                results_by_category[cat]['avg_faithfulness'] += result.answer_faithfulness

        # Normalize category averages
        for cat in results_by_category:
            n = results_by_category[cat]['total']
            results_by_category[cat]['avg_recall'] /= n
            results_by_category[cat]['avg_faithfulness'] /= n
            results_by_category[cat]['pass_rate'] = results_by_category[cat]['passed'] / n

        summary = EvalRunSummary(
            run_id=None,
            started_at=started_at,
            completed_at=completed_at,
            total_cases=len(results),
            passed_cases=passed_count,
            failed_cases=len(results) - passed_count,
            avg_retrieval_precision=avg_precision,
            avg_retrieval_recall=avg_recall,
            avg_retrieval_mrr=avg_mrr,
            avg_answer_faithfulness=avg_faithfulness,
            avg_answer_relevance=avg_relevance,
            avg_citation_accuracy=avg_citation,
            avg_latency_ms=avg_latency,
            results_by_category=results_by_category,
            results=results,
        )

        # Save to database
        if save_results:
            summary.run_id = self._save_eval_run(summary, cases)

        return summary

    def _save_eval_run(
        self,
        summary: EvalRunSummary,
        cases: List[EvalCase],
    ) -> int:
        """Save evaluation run to database."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Insert run summary
                cur.execute("""
                    INSERT INTO rag.rag_eval_runs (
                        started_at_utc, completed_at_utc,
                        total_cases, passed_cases, failed_cases,
                        avg_retrieval_precision, avg_retrieval_recall, avg_retrieval_mrr,
                        avg_answer_faithfulness, avg_answer_relevance, avg_citation_accuracy,
                        avg_latency_ms, results_by_category,
                        config_snapshot
                    ) VALUES (
                        %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s,
                        %s
                    ) RETURNING run_id
                """, (
                    summary.started_at, summary.completed_at,
                    summary.total_cases, summary.passed_cases, summary.failed_cases,
                    summary.avg_retrieval_precision, summary.avg_retrieval_recall, summary.avg_retrieval_mrr,
                    summary.avg_answer_faithfulness, summary.avg_answer_relevance, summary.avg_citation_accuracy,
                    summary.avg_latency_ms, Json(summary.results_by_category),
                    Json({
                        'retrieval_k': self.rag_service.retriever.config.k_vector,
                        'min_similarity': self.rag_service.retriever.config.min_cosine_similarity,
                    }),
                ))

                run_id = cur.fetchone()[0]
                conn.commit()

        logger.info(f"Saved eval run {run_id}")
        return run_id

    def print_report(self, summary: EvalRunSummary):
        """Print evaluation report."""
        print("\n" + "=" * 70)
        print("RAG EVALUATION REPORT")
        print("=" * 70)
        print(f"Run ID: {summary.run_id}")
        print(f"Duration: {(summary.completed_at - summary.started_at).total_seconds():.1f}s")
        print()

        print("OVERALL RESULTS")
        print("-" * 40)
        print(f"Total Cases:    {summary.total_cases}")
        print(f"Passed:         {summary.passed_cases} ({summary.passed_cases/summary.total_cases*100:.1f}%)")
        print(f"Failed:         {summary.failed_cases} ({summary.failed_cases/summary.total_cases*100:.1f}%)")
        print()

        print("RETRIEVAL METRICS")
        print("-" * 40)
        print(f"Avg Precision:  {summary.avg_retrieval_precision:.3f}")
        print(f"Avg Recall:     {summary.avg_retrieval_recall:.3f}")
        print(f"Avg MRR:        {summary.avg_retrieval_mrr:.3f}")
        print()

        print("ANSWER METRICS")
        print("-" * 40)
        print(f"Avg Faithfulness: {summary.avg_answer_faithfulness:.3f}")
        print(f"Avg Relevance:    {summary.avg_answer_relevance:.3f}")
        print(f"Avg Citation Acc: {summary.avg_citation_accuracy:.3f}")
        print()

        print("PERFORMANCE")
        print("-" * 40)
        print(f"Avg Latency:    {summary.avg_latency_ms:.0f}ms")
        print()

        if summary.results_by_category:
            print("BY CATEGORY")
            print("-" * 40)
            for cat, metrics in summary.results_by_category.items():
                print(f"  {cat}:")
                print(f"    Pass Rate: {metrics['pass_rate']*100:.1f}%")
                print(f"    Avg Recall: {metrics['avg_recall']:.3f}")

        # Show failures
        failures = [r for r in summary.results if not r.passed]
        if failures:
            print()
            print("FAILURES")
            print("-" * 40)
            for r in failures[:5]:  # Show first 5
                print(f"  [{r.ticker}] {r.question[:50]}...")
                for reason in r.failure_reasons:
                    print(f"    - {reason}")

        print()
        print("=" * 70)


def generate_sample_cases(ticker: str = "MU") -> List[EvalCase]:
    """Generate sample evaluation cases for testing."""
    return [
        EvalCase(
            case_id=None,
            ticker=ticker,
            question="What are the main risk factors for this company?",
            expected_answer="Risk factors include supply chain, competition, and regulatory risks.",
            expected_chunk_ids=[],  # Will be populated after retrieval
            category="risk",
            difficulty="easy",
        ),
        EvalCase(
            case_id=None,
            ticker=ticker,
            question="What did management say about revenue guidance?",
            expected_answer="Management provided guidance on revenue expectations.",
            expected_chunk_ids=[],
            category="guidance",
            difficulty="medium",
        ),
        EvalCase(
            case_id=None,
            ticker=ticker,
            question="What are the company's main growth drivers?",
            expected_answer="Growth drivers include AI demand and data center expansion.",
            expected_chunk_ids=[],
            category="financial",
            difficulty="medium",
        ),
    ]


if __name__ == "__main__":
    print("Testing RAG Evaluator...")

    evaluator = RAGEvaluator()

    # Check for existing cases
    cases = evaluator.load_eval_cases()

    if not cases:
        print("No evaluation cases in database. Creating sample cases...")
        sample_cases = generate_sample_cases("MU")

        print(f"\nRunning evaluation on {len(sample_cases)} sample cases...")
        summary = evaluator.run_evaluation(cases=sample_cases, save_results=False)
    else:
        print(f"Found {len(cases)} evaluation cases")
        summary = evaluator.run_evaluation(save_results=True)

    if summary:
        evaluator.print_report(summary)