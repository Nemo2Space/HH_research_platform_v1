"""
Alpha Platform - Committee Coordinator

Orchestrates multi-agent debate and aggregates decisions.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

from src.committee.agents import (
    BaseAgent, AgentVote, CommitteeDecision,
    FundamentalAgent, SentimentAgent, TechnicalAgent, ValuationAgent
)
from src.db.repository import Repository
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Default agent weights
DEFAULT_WEIGHTS = {
    'fundamental': 0.30,
    'sentiment': 0.20,
    'technical': 0.25,
    'valuation': 0.25,
}


class CommitteeCoordinator:
    """Coordinates multi-agent analysis and aggregates decisions."""

    def __init__(self, llm_client=None, repository: Optional[Repository] = None,
                 weights: Optional[Dict[str, float]] = None):
        self.llm = llm_client
        self.repo = repository or Repository()
        self.weights = weights or DEFAULT_WEIGHTS

        # Initialize agents
        self.agents = {
            'fundamental': FundamentalAgent(llm_client),
            'sentiment': SentimentAgent(llm_client),
            'technical': TechnicalAgent(llm_client),
            'valuation': ValuationAgent(llm_client),
        }

        self.agent_order = ['fundamental', 'sentiment', 'technical', 'valuation']

    def gather_data(self, ticker: str) -> Dict[str, Any]:
        """Gather all data needed for analysis."""
        import pandas as pd

        data = {'ticker': ticker}

        # Fundamentals
        query = "SELECT * FROM fundamentals WHERE ticker = %(ticker)s ORDER BY date DESC LIMIT 1"
        df = pd.read_sql(query, self.repo.engine, params={"ticker": ticker})
        if len(df) > 0:
            data['fundamentals'] = df.iloc[0].to_dict()

        # Screener scores
        query = "SELECT * FROM screener_scores WHERE ticker = %(ticker)s ORDER BY date DESC LIMIT 1"
        df = pd.read_sql(query, self.repo.engine, params={"ticker": ticker})
        if len(df) > 0:
            row = df.iloc[0]
            data['sentiment_score'] = row.get('sentiment_score')
            data['technical_score'] = row.get('technical_score')
            data['fundamental_score'] = row.get('fundamental_score')
            data['gap_score'] = row.get('gap_score', 50)
            data['article_count'] = row.get('article_count', 0)

        # Technical details
        from src.screener.technicals import TechnicalAnalyzer
        tech = TechnicalAnalyzer(self.repo)
        tech_data = tech.analyze_ticker(ticker)
        data.update({
            'rsi': tech_data.get('rsi', 50),
            'trend': tech_data.get('trend', 'neutral'),
            'momentum_5d': tech_data.get('momentum_5d', 0),
            'macd_signal': tech_data.get('macd_signal', 'neutral'),
        })

        # Analyst data
        query = "SELECT * FROM price_targets WHERE ticker = %(ticker)s ORDER BY date DESC LIMIT 1"
        df = pd.read_sql(query, self.repo.engine, params={"ticker": ticker})
        if len(df) > 0:
            row = df.iloc[0]
            data['target_upside_pct'] = row.get('target_upside_pct', 0)

        query = "SELECT * FROM analyst_ratings WHERE ticker = %(ticker)s ORDER BY date DESC LIMIT 1"
        df = pd.read_sql(query, self.repo.engine, params={"ticker": ticker})
        if len(df) > 0:
            row = df.iloc[0]
            data['analyst_positivity'] = row.get('analyst_positivity', 50)

        return data

    def run_agents(self, ticker: str, data: Dict[str, Any]) -> List[AgentVote]:
        """Run all agents and collect votes."""
        votes = []

        for role in self.agent_order:
            agent = self.agents.get(role)
            if not agent:
                continue

            try:
                vote = agent.analyze(ticker, data)
                votes.append(vote)
                logger.info(f"{ticker}: {role} agent voted buy_prob={vote.buy_prob:.2f}, alpha={vote.expected_alpha_bps:.0f}bps")
            except Exception as e:
                logger.error(f"{ticker}: {role} agent failed - {e}")

        return votes

    def aggregate_votes(self, ticker: str, votes: List[AgentVote]) -> CommitteeDecision:
        """Aggregate agent votes into final decision."""
        asof = datetime.now().strftime("%Y-%m-%d")

        if not votes:
            return CommitteeDecision(
                ticker=ticker,
                asof=asof,
                verdict="HOLD",
                expected_alpha_bps=0,
                confidence=0.1,
                horizon_days=63,
                conviction=10,
                rationale="No agent votes available",
                risks=["Insufficient data"],
                votes=[]
            )

        # Weighted aggregation
        total_weight = 0
        weighted_alpha = 0
        weighted_buy_prob = 0
        weighted_confidence = 0
        all_risks = []
        rationales = []
        horizons = []

        for vote in votes:
            weight = self.weights.get(vote.role, 0.25) * vote.confidence
            total_weight += weight

            weighted_alpha += vote.expected_alpha_bps * weight
            weighted_buy_prob += vote.buy_prob * weight
            weighted_confidence += vote.confidence * weight

            horizons.append(vote.horizon_days)
            all_risks.extend(vote.risks[:2])
            if vote.rationale:
                rationales.append(f"{vote.role.capitalize()}: {vote.rationale}")

        if total_weight > 0:
            final_alpha = weighted_alpha / total_weight
            final_buy_prob = weighted_buy_prob / total_weight
            avg_confidence = weighted_confidence / total_weight
        else:
            final_alpha = 0
            final_buy_prob = 0.5
            avg_confidence = 0.3

        # Adjust confidence for agreement
        alphas = [v.expected_alpha_bps for v in votes]
        alpha_std = float(np.std(alphas)) if len(alphas) > 1 else 0

        if alpha_std < 30:  # Good agreement
            avg_confidence = min(1.0, avg_confidence * 1.15)
        elif alpha_std > 100:  # High disagreement
            avg_confidence = avg_confidence * 0.85

        # Determine verdict
        if final_buy_prob >= 0.65 and final_alpha >= 50:
            verdict = "BUY"
        elif final_buy_prob >= 0.55 and final_alpha >= 20:
            verdict = "WEAK BUY"
        elif final_buy_prob <= 0.35 and final_alpha <= -50:
            verdict = "SELL"
        elif final_buy_prob <= 0.45 and final_alpha <= -20:
            verdict = "WEAK SELL"
        else:
            verdict = "HOLD"

        # Conviction (0-100)
        conviction = int(avg_confidence * 100)

        # Horizon
        horizon_days = int(np.median(horizons)) if horizons else 63

        # Rationale
        rationale = " | ".join(rationales) if rationales else "Committee analysis complete"

        # Unique risks
        unique_risks = list(dict.fromkeys(all_risks))[:5]

        logger.info(f"{ticker}: Committee verdict={verdict}, alpha={final_alpha:.0f}bps, conviction={conviction}%")

        return CommitteeDecision(
            ticker=ticker,
            asof=asof,
            verdict=verdict,
            expected_alpha_bps=round(final_alpha, 1),
            confidence=round(avg_confidence, 2),
            horizon_days=horizon_days,
            conviction=conviction,
            rationale=rationale,
            risks=unique_risks,
            votes=votes
        )

    def analyze_ticker(self, ticker: str) -> CommitteeDecision:
        """Run full committee analysis on a ticker."""
        logger.info(f"Starting committee analysis for {ticker}")

        # Gather data
        data = self.gather_data(ticker)

        # Run agents
        votes = self.run_agents(ticker, data)

        # Aggregate
        decision = self.aggregate_votes(ticker, votes)

        # Save to database
        self._save_decision(decision)

        return decision

    def _save_decision(self, decision: CommitteeDecision):
        """Save decision to database."""
        try:
            from src.db.connection import get_connection

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

            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Save main decision
                    cur.execute("""
                                INSERT INTO committee_decisions (ticker, date, verdict, conviction, expected_alpha_bps,
                                                                 horizon_days, rationale)
                                VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (ticker, date) DO
                                UPDATE SET
                                    verdict = EXCLUDED.verdict,
                                    conviction = EXCLUDED.conviction,
                                    expected_alpha_bps = EXCLUDED.expected_alpha_bps,
                                    horizon_days = EXCLUDED.horizon_days,
                                    rationale = EXCLUDED.rationale
                                """, (
                                    decision.ticker,
                                    decision.asof,
                                    decision.verdict,
                                    to_python(decision.conviction),
                                    to_python(decision.expected_alpha_bps),
                                    to_python(decision.horizon_days),
                                    decision.rationale
                                ))

                    # Save individual votes
                    for vote in decision.votes:
                        cur.execute("""
                                    INSERT INTO agent_votes (ticker, date, agent_role, buy_prob, expected_alpha_bps,
                                                             confidence, rationale)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (ticker, date, agent_role) DO
                                    UPDATE SET
                                        buy_prob = EXCLUDED.buy_prob,
                                        expected_alpha_bps = EXCLUDED.expected_alpha_bps,
                                        confidence = EXCLUDED.confidence,
                                        rationale = EXCLUDED.rationale
                                    """, (
                                        decision.ticker,
                                        decision.asof,
                                        vote.role,
                                        to_python(vote.buy_prob),
                                        to_python(vote.expected_alpha_bps),
                                        to_python(vote.confidence),
                                        vote.rationale
                                    ))

            logger.info(f"{decision.ticker}: Decision saved to database")

        except Exception as e:
            logger.error(f"Failed to save decision: {e}")


    def analyze_universe(self, progress_callback=None) -> Dict[str, CommitteeDecision]:
        """Run committee analysis on all tickers."""
        tickers = self.repo.get_universe()
        total = len(tickers)

        results = {}

        for i, ticker in enumerate(tickers):
            if progress_callback:
                progress_callback(i + 1, total, ticker)

            try:
                decision = self.analyze_ticker(ticker)
                results[ticker] = decision
            except Exception as e:
                logger.error(f"{ticker}: Committee analysis failed - {e}")

        return results