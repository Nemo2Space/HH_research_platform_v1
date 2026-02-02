"""
RAG Memory System - Phase 4

Case-based reasoning using similar historical setups.
CRITICAL: Setup cards are leakage-safe - only contain data known at T0.

Location: src/ml/rag_memory.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SetupCard:
    """Structured snapshot of a trading setup at decision time (T0)."""
    card_id: str
    ticker: str
    setup_date: date

    # Features at T0
    sentiment_score: float
    fundamental_score: float
    technical_score: float
    options_flow_score: float
    short_squeeze_score: float
    total_score: float

    # Market context at T0
    sector: str
    vix_level: float
    regime_score: float

    # Signal at T0
    signal_type: str
    ml_probability: float
    expected_value: float

    # Trade plan at T0
    entry_price: float
    planned_horizon: int

    # Outcome (attached AFTER - never used for retrieval)
    outcome: Optional[str] = None  # WIN, LOSS, SCRATCH
    actual_return: Optional[float] = None
    exit_reason: Optional[str] = None

    def to_feature_vector(self) -> np.ndarray:
        """Convert to numeric vector for similarity search."""
        return np.array([
            self.sentiment_score, self.fundamental_score, self.technical_score,
            self.options_flow_score, self.short_squeeze_score, self.total_score,
            self.vix_level, self.regime_score, self.ml_probability * 100,
        ])

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker, 'setup_date': str(self.setup_date),
            'total_score': self.total_score, 'ml_probability': self.ml_probability,
            'outcome': self.outcome, 'actual_return': self.actual_return
        }


@dataclass
class SimilarSetupResult:
    """Result of finding similar historical setups."""
    similar_cards: List[SetupCard]
    similarity_scores: List[float]
    closed_count: int
    win_rate: float
    avg_return: float
    best_return: float
    worst_return: float
    common_loss_reasons: Dict[str, int]

    def get_insight_summary(self) -> str:
        if self.closed_count == 0:
            return "No similar historical setups with outcomes found."

        summary = f"Found {self.closed_count} similar setups: "
        summary += f"Win Rate={self.win_rate:.1%}, Avg Return={self.avg_return:+.2f}%"

        if self.win_rate >= 0.65:
            summary += " ✅ Strong edge"
        elif self.win_rate < 0.50:
            summary += " ❌ Weak edge"

        return summary


class RAGMemoryStore:
    """
    RAG-style memory for similar historical setups.

    Key principle: Retrieve by features, evaluate by outcomes.
    Only uses T0 data for retrieval (leakage-safe).
    """

    # Cost threshold (0.15% round trip)
    COST_PCT = 0.15

    def __init__(self):
        self._cards: List[SetupCard] = []
        self._feature_matrix: Optional[np.ndarray] = None
        self._scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self._fitted = False

    def store_card(self, card: SetupCard):
        """Store a new setup card."""
        self._cards.append(card)
        self._fitted = False  # Need to refit

    def update_outcome(self, ticker: str, setup_date: date,
                       outcome: str, actual_return: float, exit_reason: str):
        """Update outcome for an existing card."""
        for card in self._cards:
            if card.ticker == ticker and card.setup_date == setup_date:
                card.outcome = outcome
                card.actual_return = actual_return
                card.exit_reason = exit_reason
                break

    def find_similar(self, query_card: SetupCard, top_k: int = 20) -> SimilarSetupResult:
        """Find similar historical setups using feature similarity."""
        if not self._cards:
            return self._empty_result()

        if not SKLEARN_AVAILABLE:
            return self._empty_result()

        # Build feature matrix if needed
        if not self._fitted:
            self._build_feature_matrix()

        if self._feature_matrix is None or len(self._feature_matrix) == 0:
            return self._empty_result()

        # Query vector
        query_vec = query_card.to_feature_vector().reshape(1, -1)
        query_scaled = self._scaler.transform(query_vec)

        # Similarity search
        sims = cosine_similarity(query_scaled, self._feature_matrix)[0]

        # Top K indices
        top_idx = np.argsort(sims)[-top_k:][::-1]

        # Get cards and scores
        similar = [self._cards[i] for i in top_idx if i < len(self._cards)]
        scores = [float(sims[i]) for i in top_idx if i < len(self._cards)]

        # Stats from closed trades only
        closed = [c for c in similar if c.outcome]
        if closed:
            wins = [c for c in closed if c.outcome == 'WIN']
            win_rate = len(wins) / len(closed)
            returns = [c.actual_return for c in closed if c.actual_return is not None]
            avg_ret = np.mean(returns) if returns else 0
            best = max(returns) if returns else 0
            worst = min(returns) if returns else 0

            loss_reasons = {}
            for c in closed:
                if c.outcome == 'LOSS' and c.exit_reason:
                    loss_reasons[c.exit_reason] = loss_reasons.get(c.exit_reason, 0) + 1
        else:
            win_rate = avg_ret = best = worst = 0
            loss_reasons = {}

        return SimilarSetupResult(
            similar_cards=similar, similarity_scores=scores,
            closed_count=len(closed), win_rate=win_rate,
            avg_return=avg_ret, best_return=best, worst_return=worst,
            common_loss_reasons=loss_reasons
        )

    def _build_feature_matrix(self):
        """Build and fit feature matrix for all cards."""
        if not self._cards:
            self._feature_matrix = None
            return

        vectors = np.array([c.to_feature_vector() for c in self._cards])
        self._scaler.fit(vectors)
        self._feature_matrix = self._scaler.transform(vectors)
        self._fitted = True
        logger.info(f"Built feature matrix with {len(self._cards)} cards")

    def _empty_result(self) -> SimilarSetupResult:
        return SimilarSetupResult([], [], 0, 0, 0, 0, 0, {})

    def load_from_db(self, engine=None) -> int:
        """
        Load setup cards from historical_scores table.

        Joins with screener_scores to get additional columns when available.
        Uses cost-adjusted returns to determine WIN/LOSS.
        """
        if engine is None:
            from src.ml.db_helper import get_engine
            engine = get_engine()

        # Query historical_scores with screener_scores join for additional data
        query = """
            SELECT 
                h.ticker,
                h.score_date,
                h.sector,
                COALESCE(h.sentiment, 50) as sentiment_score,
                COALESCE(h.fundamental_score, 50) as fundamental_score,
                COALESCE(h.growth_score, 50) as growth_score,
                COALESCE(h.total_score, 50) as total_score,
COALESCE(CASE WHEN h.gap_score ~ '^[0-9.\\-]+$' THEN h.gap_score::numeric ELSE NULL END, 50) as gap_score,
                h.signal_type,
                h.op_price as entry_price,
                h.return_1d,
                h.return_5d,
                h.return_10d,
                h.return_20d,
                -- Additional columns from screener_scores if available
                COALESCE(s.technical_score, 50) as technical_score,
                COALESCE(s.options_flow_score, 50) as options_flow_score,
                COALESCE(s.short_squeeze_score, 30) as short_squeeze_score
            FROM historical_scores h
            LEFT JOIN screener_scores s 
                ON h.ticker = s.ticker 
                AND h.score_date = s.date
            WHERE h.return_5d IS NOT NULL 
              AND h.op_price IS NOT NULL 
              AND h.op_price > 0
            ORDER BY h.score_date DESC
        """

        try:
            df = pd.read_sql(query, engine)
        except Exception as e:
            logger.error(f"Failed to load from DB: {e}")
            return 0

        if df.empty:
            logger.warning("No historical data found")
            return 0

        logger.info(f"Loading {len(df)} historical records")

        # Clear existing cards
        self._cards = []

        for _, row in df.iterrows():
            # Cost-adjusted return (0.15% round trip cost)
            return_5d = float(row['return_5d'])
            net_return = return_5d - self.COST_PCT

            # Determine outcome based on cost-adjusted return
            if net_return > 0:
                outcome = 'WIN'
            elif net_return < -1.0:  # Lost more than 1%
                outcome = 'LOSS'
            else:
                outcome = 'SCRATCH'  # Roughly flat after costs

            # Determine exit reason based on return pattern
            if outcome == 'WIN':
                exit_reason = 'TARGET'
            elif outcome == 'LOSS':
                exit_reason = 'STOP' if return_5d < -3 else 'TIME'
            else:
                exit_reason = 'TIME'

            card = SetupCard(
                card_id=f"{row['ticker']}_{row['score_date']}",
                ticker=row['ticker'],
                setup_date=row['score_date'] if isinstance(row['score_date'], date) else pd.to_datetime(row['score_date']).date(),
                sentiment_score=float(row['sentiment_score']),
                fundamental_score=float(row['fundamental_score']),
                technical_score=float(row['technical_score']),
                options_flow_score=float(row['options_flow_score']),
                short_squeeze_score=float(row['short_squeeze_score']),
                total_score=float(row['total_score']),
                sector=str(row['sector']) if row['sector'] else 'Unknown',
                vix_level=20,  # Could enhance by storing VIX history
                regime_score=50,  # Could enhance with regime data
                signal_type=row['signal_type'] or 'HOLD',
                ml_probability=0.5,  # Not available for historical
                expected_value=0,
                entry_price=float(row['entry_price']),
                planned_horizon=5,
                outcome=outcome,
                actual_return=return_5d,
                exit_reason=exit_reason
            )
            self._cards.append(card)

        # Build feature matrix
        self._build_feature_matrix()

        # Log stats
        closed = [c for c in self._cards if c.outcome]
        wins = [c for c in closed if c.outcome == 'WIN']
        win_rate = len(wins) / len(closed) if closed else 0

        logger.info(f"Loaded {len(self._cards)} setup cards: {len(wins)} wins, {len(closed)-len(wins)} losses ({win_rate:.1%} win rate)")

        return len(self._cards)

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        closed = [c for c in self._cards if c.outcome]
        wins = [c for c in closed if c.outcome == 'WIN']

        returns = [c.actual_return for c in closed if c.actual_return is not None]

        return {
            'total_cards': len(self._cards),
            'closed_cards': len(closed),
            'win_count': len(wins),
            'loss_count': len(closed) - len(wins),
            'win_rate': len(wins) / len(closed) if closed else 0,
            'avg_return': np.mean(returns) if returns else 0,
            'best_return': max(returns) if returns else 0,
            'worst_return': min(returns) if returns else 0,
        }


# Convenience function to populate memory
def populate_rag_memory() -> RAGMemoryStore:
    """Create and populate RAG memory from database."""
    store = RAGMemoryStore()
    count = store.load_from_db()
    logger.info(f"Populated RAG memory with {count} setup cards")
    return store


if __name__ == "__main__":
    # Test loading
    store = RAGMemoryStore()
    count = store.load_from_db()

    print(f"\nLoaded {count} setup cards")
    print("\nStats:", store.get_stats())