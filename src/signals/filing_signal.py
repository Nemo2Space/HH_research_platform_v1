"""
Filing Signal - Signal Component from SEC Filings
=================================================

Generates a signal score (0-100) from extracted SEC filing facts.

Factors considered:
- Guidance direction (raised/lowered)
- Risk severity and count
- China exposure level
- Material litigation
- AI demand mentions

Usage:
    from src.signals.filing_signal import get_filing_signal, FilingSignal

    signal = get_filing_signal("MU")
    print(signal.score)  # 0-100
    print(signal.factors)

Integration with unified_scorer.py:
    filing_signal = get_filing_signal(ticker)
    if filing_signal:
        scores['filing'] = filing_signal.score

Author: HH Research Platform
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from src.db.connection import get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FilingSignalFactors:
    """Individual factors contributing to filing signal."""
    guidance_score: float = 50.0  # 0-100
    risk_score: float = 50.0  # 0-100 (inverted - more risk = lower)
    litigation_score: float = 50.0  # 0-100
    china_score: float = 50.0  # 0-100
    ai_demand_score: float = 50.0  # 0-100

    guidance_direction: Optional[str] = None
    risk_count: int = 0
    has_litigation: bool = False
    china_exposure: Optional[str] = None
    ai_mentions: int = 0


@dataclass
class FilingSignal:
    """Filing-based signal result."""
    ticker: str
    score: float  # 0-100, composite score

    # Components
    factors: FilingSignalFactors = field(default_factory=FilingSignalFactors)

    # Weights used
    weights: Dict[str, float] = field(default_factory=dict)

    # Data quality
    has_filing_data: bool = False
    has_transcript_data: bool = False
    data_freshness_days: int = 0

    # Explanations
    bullish_signals: List[str] = field(default_factory=list)
    bearish_signals: List[str] = field(default_factory=list)

    # Metadata
    filings_analyzed: int = 0
    computed_at: datetime = field(default_factory=datetime.utcnow)


# Default weights
DEFAULT_WEIGHTS = {
    'guidance': 0.30,
    'risk': 0.25,
    'litigation': 0.15,
    'china': 0.15,
    'ai_demand': 0.15,
}

# Risk theme severity mapping
RISK_SEVERITY = {
    'supply_chain': 0.7,
    'china_exposure': 0.8,
    'pricing_pressure': 0.6,
    'demand_weakness': 0.9,
    'regulation': 0.5,
    'competition': 0.5,
    'customer_concentration': 0.7,
    'litigation': 0.6,
    'cybersecurity': 0.5,
    'geopolitical': 0.6,
    'inventory': 0.4,
    'labor': 0.3,
    'currency': 0.3,
    'interest_rates': 0.4,
    'inflation': 0.4,
    'environmental': 0.3,
}


def get_filing_signal(
        ticker: str,
        weights: Dict[str, float] = None,
        max_age_days: int = 90,
) -> Optional[FilingSignal]:
    """
    Calculate filing-based signal for a ticker.

    Args:
        ticker: Stock ticker
        weights: Custom weights for factors (default: DEFAULT_WEIGHTS)
        max_age_days: Max age of filing data to consider

    Returns:
        FilingSignal or None if no data
    """
    weights = weights or DEFAULT_WEIGHTS

    factors = FilingSignalFactors()
    bullish = []
    bearish = []

    has_filing = False
    has_transcript = False
    filings_count = 0
    freshness_days = max_age_days

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Get latest filing facts
            cur.execute("""
                SELECT 
                    key_risks,
                    risk_severity,
                    china_exposure,
                    material_litigation,
                    extracted_at_utc
                FROM rag.filing_facts
                WHERE ticker = %s
                  AND extracted_at_utc > NOW() - INTERVAL '%s days'
                ORDER BY extracted_at_utc DESC
                LIMIT 1
            """, (ticker, max_age_days))

            filing_row = cur.fetchone()

            if filing_row:
                has_filing = True
                filings_count += 1

                key_risks = filing_row[0] or []
                risk_severity = filing_row[1] or {}
                china_exposure = filing_row[2]
                material_litigation = filing_row[3]
                extracted_at = filing_row[4]

                # Calculate freshness
                if extracted_at:
                    freshness_days = (datetime.utcnow() - extracted_at.replace(tzinfo=None)).days

                # Risk score (inverted - more severe risks = lower score)
                if key_risks:
                    factors.risk_count = len(key_risks)
                    total_severity = sum(RISK_SEVERITY.get(r, 0.5) for r in key_risks)
                    avg_severity = total_severity / len(key_risks)
                    factors.risk_score = max(0, 100 - (avg_severity * 60) - (len(key_risks) * 5))

                    if factors.risk_score < 40:
                        bearish.append(f"High risk exposure ({len(key_risks)} major risks)")
                    elif factors.risk_score > 70:
                        bullish.append("Low risk profile")

                # China exposure score
                if china_exposure:
                    factors.china_exposure = china_exposure
                    # Simple heuristic - presence of china exposure text
                    china_lower = china_exposure.lower()
                    if 'significant' in china_lower or 'material' in china_lower:
                        factors.china_score = 30
                        bearish.append("Significant China exposure")
                    elif 'limited' in china_lower or 'minimal' in china_lower:
                        factors.china_score = 80
                        bullish.append("Limited China exposure")
                    else:
                        factors.china_score = 50

                # Litigation score
                factors.has_litigation = material_litigation or False
                if material_litigation:
                    factors.litigation_score = 25
                    bearish.append("Material litigation disclosed")
                else:
                    factors.litigation_score = 75
                    bullish.append("No material litigation")

            # Get transcript facts for guidance and AI demand
            cur.execute("""
                SELECT 
                    guidance_direction,
                    demand_tone,
                    ai_mentions_count,
                    ai_sentiment,
                    extracted_at_utc
                FROM rag.transcript_facts
                WHERE ticker = %s
                  AND extracted_at_utc > NOW() - INTERVAL '%s days'
                ORDER BY extracted_at_utc DESC
                LIMIT 1
            """, (ticker, max_age_days))

            transcript_row = cur.fetchone()

            if transcript_row:
                has_transcript = True
                filings_count += 1

                guidance_dir = transcript_row[0]
                demand_tone = transcript_row[1]
                ai_mentions = transcript_row[2] or 0
                ai_sentiment = transcript_row[3]

                # Guidance score
                factors.guidance_direction = guidance_dir
                if guidance_dir:
                    guidance_scores = {
                        'raised': 90,
                        'initiated': 75,
                        'maintained': 55,
                        'lowered': 20,
                        'withdrawn': 10,
                    }
                    factors.guidance_score = guidance_scores.get(guidance_dir, 50)

                    if guidance_dir == 'raised':
                        bullish.append("Management raised guidance")
                    elif guidance_dir == 'lowered':
                        bearish.append("Management lowered guidance")
                    elif guidance_dir == 'withdrawn':
                        bearish.append("Guidance withdrawn")

                # AI demand score
                factors.ai_mentions = ai_mentions
                if ai_mentions > 0:
                    # More AI mentions = more bullish (for tech stocks)
                    factors.ai_demand_score = min(100, 50 + (ai_mentions * 3))

                    if ai_sentiment == 'positive':
                        factors.ai_demand_score = min(100, factors.ai_demand_score + 15)
                        bullish.append(f"Strong AI demand narrative ({ai_mentions} mentions)")
                    elif ai_sentiment == 'negative':
                        factors.ai_demand_score = max(0, factors.ai_demand_score - 20)

                # Adjust for demand tone
                if demand_tone:
                    tone_adjustments = {
                        'bullish': 10,
                        'neutral': 0,
                        'cautious': -10,
                        'bearish': -20,
                    }
                    adjustment = tone_adjustments.get(demand_tone, 0)
                    factors.guidance_score = max(0, min(100, factors.guidance_score + adjustment))

    # If no data, return None
    if not has_filing and not has_transcript:
        logger.debug(f"No filing data for {ticker}")
        return None

    # Calculate composite score
    composite = (
            factors.guidance_score * weights['guidance'] +
            factors.risk_score * weights['risk'] +
            factors.litigation_score * weights['litigation'] +
            factors.china_score * weights['china'] +
            factors.ai_demand_score * weights['ai_demand']
    )

    signal = FilingSignal(
        ticker=ticker,
        score=round(composite, 1),
        factors=factors,
        weights=weights,
        has_filing_data=has_filing,
        has_transcript_data=has_transcript,
        data_freshness_days=freshness_days,
        bullish_signals=bullish,
        bearish_signals=bearish,
        filings_analyzed=filings_count,
    )

    logger.debug(f"Filing signal for {ticker}: {signal.score:.1f}")

    return signal


def get_filing_signal_for_scorer(ticker: str) -> Optional[float]:
    """
    Simple interface for unified_scorer.py integration.

    Returns score 0-100 or None if no data.
    """
    signal = get_filing_signal(ticker)
    return signal.score if signal else None


def get_filing_insights(ticker: str) -> Dict[str, Any]:
    """
    Get detailed filing insights for display in UI.
    """
    signal = get_filing_signal(ticker)

    if not signal:
        return {
            "available": False,
            "ticker": ticker,
            "message": "No SEC filing data available",
        }

    return {
        "available": True,
        "ticker": ticker,
        "score": signal.score,
        "score_label": _score_to_label(signal.score),

        "factors": {
            "guidance": {
                "score": signal.factors.guidance_score,
                "direction": signal.factors.guidance_direction,
                "weight": signal.weights['guidance'],
            },
            "risk": {
                "score": signal.factors.risk_score,
                "count": signal.factors.risk_count,
                "weight": signal.weights['risk'],
            },
            "litigation": {
                "score": signal.factors.litigation_score,
                "has_material": signal.factors.has_litigation,
                "weight": signal.weights['litigation'],
            },
            "china": {
                "score": signal.factors.china_score,
                "exposure": signal.factors.china_exposure,
                "weight": signal.weights['china'],
            },
            "ai_demand": {
                "score": signal.factors.ai_demand_score,
                "mentions": signal.factors.ai_mentions,
                "weight": signal.weights['ai_demand'],
            },
        },

        "bullish_signals": signal.bullish_signals,
        "bearish_signals": signal.bearish_signals,

        "data_quality": {
            "has_filing": signal.has_filing_data,
            "has_transcript": signal.has_transcript_data,
            "freshness_days": signal.data_freshness_days,
            "filings_analyzed": signal.filings_analyzed,
        },
    }


def _score_to_label(score: float) -> str:
    """Convert score to human label."""
    if score >= 80:
        return "Very Bullish"
    elif score >= 65:
        return "Bullish"
    elif score >= 50:
        return "Neutral"
    elif score >= 35:
        return "Bearish"
    else:
        return "Very Bearish"


if __name__ == "__main__":
    print("Testing Filing Signal...")

    signal = get_filing_signal("MU")

    if signal:
        print(f"\n{'=' * 50}")
        print(f"FILING SIGNAL: {signal.ticker}")
        print(f"{'=' * 50}")
        print(f"Score: {signal.score:.1f}/100 ({_score_to_label(signal.score)})")
        print(f"\nFactors:")
        print(f"  Guidance: {signal.factors.guidance_score:.0f} ({signal.factors.guidance_direction})")
        print(f"  Risk: {signal.factors.risk_score:.0f} ({signal.factors.risk_count} risks)")
        print(f"  Litigation: {signal.factors.litigation_score:.0f}")
        print(f"  China: {signal.factors.china_score:.0f}")
        print(f"  AI Demand: {signal.factors.ai_demand_score:.0f} ({signal.factors.ai_mentions} mentions)")

        if signal.bullish_signals:
            print(f"\n✅ Bullish:")
            for b in signal.bullish_signals:
                print(f"   • {b}")

        if signal.bearish_signals:
            print(f"\n⚠️ Bearish:")
            for b in signal.bearish_signals:
                print(f"   • {b}")
    else:
        print("No filing data available for MU")