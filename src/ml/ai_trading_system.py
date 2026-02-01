"""
AI Trading System - Main Orchestrator

Location: src/ml/ai_trading_system.py
"""

import os
import sys
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

# Add project root to path
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import components
from src.ml.signal_predictor import MLSignalPredictor
from src.ml.decision_layer import DecisionLayer, PortfolioState, RiskLimits
from src.ml.rag_memory import RAGMemoryStore, SetupCard
from src.ml.meta_labeler import MetaLabeler
from src.ml.llm_integration import LLMIntegration
from src.ml.monitoring import RecommendationLogger, DriftDetector, PerformanceTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Complete analysis result from the AI system."""
    ticker: str
    timestamp: datetime
    ml_probability: float
    ml_ev: float
    ml_confidence: str
    ml_top_features: Dict[str, float]
    similar_count: int
    similar_win_rate: float
    similar_avg_return: float
    meta_probability: float
    combined_probability: float
    should_trade: bool
    positive_factors: List[str]
    negative_factors: List[str]
    approved: bool
    rejection_reasons: List[str]
    position_size_pct: float
    shares: int
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward: float
    summary: str
    recommendation: str
    confidence_qualifier: str
    risks: List[str]
    thesis_breakers: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)

    def get_summary(self) -> str:
        action = "✅ BUY" if self.approved else "❌ SKIP"
        return f"""
{action} {self.ticker}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ML Probability: {self.ml_probability:.1%} | EV: {self.ml_ev*100:.2f}%
Similar Setups: {self.similar_count} trades, {self.similar_win_rate:.1%} win rate
Meta Score: {self.meta_probability:.1%} | Combined: {self.combined_probability:.1%}

{self.summary}

Bullish: {', '.join(self.positive_factors[:3]) if self.positive_factors else 'None'}
Bearish: {', '.join(self.negative_factors[:3]) if self.negative_factors else 'None'}

{'Position: ' + str(self.shares) + ' shares (' + f'{self.position_size_pct:.1%}' + ')' if self.approved else 'Rejected: ' + ', '.join(self.rejection_reasons[:2])}
{'Entry: $' + f'{self.entry_price:.2f}' + ' | Stop: $' + f'{self.stop_loss:.2f}' + ' | Target: $' + f'{self.target_price:.2f}' if self.approved else ''}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


@dataclass
class SystemStatus:
    initialized: bool
    ml_model_loaded: bool
    ml_model_trained_at: Optional[datetime]
    ml_model_auc: float
    rag_memory_count: int
    meta_labeler_loaded: bool
    monitoring_enabled: bool
    total_recommendations: int
    recent_win_rate: float
    drift_alerts: List[str]
    needs_retrain: bool


class AITradingSystem:
    """Main AI Trading System that orchestrates all components."""

    def __init__(self, models_dir: str = "models", conservative_mode: bool = False,
                 enable_monitoring: bool = True):
        self.models_dir = models_dir
        self.conservative_mode = conservative_mode
        self.enable_monitoring = enable_monitoring

        self.ml_predictor: Optional[MLSignalPredictor] = None
        self.decision_layer: Optional[DecisionLayer] = None
        self.rag_memory: Optional[RAGMemoryStore] = None
        self.meta_labeler: Optional[MetaLabeler] = None
        self.llm: Optional[LLMIntegration] = None

        self.rec_logger: Optional[RecommendationLogger] = None
        self.drift_detector: Optional[DriftDetector] = None
        self.perf_tracker: Optional[PerformanceTracker] = None

        self._initialized = False
        self._portfolio: Optional[PortfolioState] = None

    def initialize(self, train_if_needed: bool = True, portfolio: PortfolioState = None) -> bool:
        logger.info("Initializing AI Trading System...")
        os.makedirs(self.models_dir, exist_ok=True)

        # ML Predictor
        self.ml_predictor = MLSignalPredictor()
        ml_path = os.path.join(self.models_dir, "signal_predictor.pkl")

        if os.path.exists(ml_path):
            try:
                self.ml_predictor.load(ml_path)
                logger.info("ML model loaded from disk")
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}")
                if train_if_needed:
                    self._train_ml_model()
        elif train_if_needed:
            self._train_ml_model()

        # Decision Layer
        if self.conservative_mode:
            limits = RiskLimits(max_position_pct=0.03, max_sector_pct=0.15,
                               max_positions=20, max_vix_for_new=30)
            self.decision_layer = DecisionLayer(limits=limits, min_ev=0.002, min_probability=0.58)
        else:
            self.decision_layer = DecisionLayer()

        # Other components
        self.rag_memory = RAGMemoryStore()
        self.meta_labeler = MetaLabeler()
        self.llm = LLMIntegration()

        # Monitoring
        if self.enable_monitoring:
            self.rec_logger = RecommendationLogger()
            self.drift_detector = DriftDetector()
            self.perf_tracker = PerformanceTracker(self.rec_logger)

        self._portfolio = portfolio or PortfolioState.empty()
        self._initialized = True
        logger.info("AI Trading System initialized successfully")
        return True

    def _train_ml_model(self):
        logger.info("Training ML model...")
        try:
            report = self.ml_predictor.train()
            self.ml_predictor.save(os.path.join(self.models_dir, "signal_predictor.pkl"))
            logger.info(f"ML model trained: AUC={report.mean_auc:.3f}")
        except Exception as e:
            logger.error(f"Failed to train ML model: {e}")

    def analyze(self, ticker: str, scores: Dict[str, float], stock_data: Dict = None) -> AnalysisResult:
        if not self._initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")

        stock_data = stock_data or {}
        scores['ticker'] = ticker
        current_price = stock_data.get('price', 100)
        sector = stock_data.get('sector', 'Unknown')
        vix = stock_data.get('vix', 20)

        # ML Prediction
        ml_prediction = self.ml_predictor.predict(scores)

        # Similar Setups
        setup_card = SetupCard(
            card_id='', ticker=ticker, setup_date=date.today(),
            sentiment_score=scores.get('sentiment_score', 50),
            fundamental_score=scores.get('fundamental_score', 50),
            technical_score=scores.get('technical_score', 50),
            options_flow_score=scores.get('options_flow_score', 50),
            short_squeeze_score=scores.get('short_squeeze_score', 50),
            total_score=scores.get('total_score', 50),
            sector=sector, vix_level=vix, regime_score=stock_data.get('regime_score', 50),
            signal_type='BUY' if ml_prediction.prob_win_5d >= 0.55 else 'HOLD',
            ml_probability=ml_prediction.prob_win_5d, expected_value=ml_prediction.ev_5d,
            entry_price=current_price, planned_horizon=5
        )
        similar_result = self.rag_memory.find_similar(setup_card)

        # Meta-Labeler
        meta_result = self.meta_labeler.predict(
            primary_prediction={'prob_win_5d': ml_prediction.prob_win_5d, 'ev_5d': ml_prediction.ev_5d},
            scores=scores,
            market_context={'vix': vix},
            similar_setups={'win_rate': similar_result.win_rate, 'count': similar_result.closed_count}
        )

        # Decision Layer
        decision = self.decision_layer.evaluate(
            ticker=ticker,
            ml_prediction={'prob_win_5d': meta_result.combined_prob, 'ev_5d': ml_prediction.ev_5d,
                          'confidence': ml_prediction.confidence},
            portfolio=self._portfolio,
            stock_data={'price': current_price, 'sector': sector,
                       'volatility': stock_data.get('volatility', 0.025),
                       'avg_volume': stock_data.get('avg_volume', 1000000)}
        )

        # Market context for LLM (includes days_to_earnings if provided)
        market_context = {
            'vix': vix,
            'days_to_earnings': stock_data.get('days_to_earnings'),  # Only upcoming, not past
        }

        # LLM Analysis
        llm_analysis = self.llm.analyze_trade(
            ticker=ticker,
            ml_prediction={'prob_win_5d': ml_prediction.prob_win_5d, 'ev_5d': ml_prediction.ev_5d,
                          'confidence': ml_prediction.confidence, 'top_features': ml_prediction.top_features},
            decision_result=decision.to_dict(),
            similar_setups={'closed_count': similar_result.closed_count, 'win_rate': similar_result.win_rate,
                           'avg_return': similar_result.avg_return,
                           'best_return': similar_result.best_return,
                           'worst_return': similar_result.worst_return},
            scores=scores,
            market_context=market_context
        )

        return AnalysisResult(
            ticker=ticker, timestamp=datetime.now(),
            ml_probability=ml_prediction.prob_win_5d, ml_ev=ml_prediction.ev_5d,
            ml_confidence=ml_prediction.confidence, ml_top_features=ml_prediction.top_features,
            similar_count=similar_result.closed_count, similar_win_rate=similar_result.win_rate,
            similar_avg_return=similar_result.avg_return,
            meta_probability=meta_result.meta_prob, combined_probability=meta_result.combined_prob,
            should_trade=meta_result.should_trade,
            positive_factors=meta_result.positive_factors, negative_factors=meta_result.negative_factors,
            approved=decision.approved, rejection_reasons=[r.value for r in decision.rejection_reasons],
            position_size_pct=decision.position_size.position_pct if decision.position_size else 0,
            shares=decision.position_size.shares if decision.position_size else 0,
            entry_price=decision.entry_price, stop_loss=decision.stop_loss, target_price=decision.target_price,
            risk_reward=decision.risk_reward_ratio,
            summary=llm_analysis.summary, recommendation=llm_analysis.recommendation,
            confidence_qualifier=llm_analysis.confidence_qualifier,
            risks=llm_analysis.risks, thesis_breakers=llm_analysis.thesis_breakers
        )

    def get_status(self) -> SystemStatus:
        ml_loaded = self.ml_predictor is not None and bool(self.ml_predictor.models)
        ml_auc = self.ml_predictor.validation_report.mean_auc if ml_loaded and self.ml_predictor.validation_report else 0.5

        return SystemStatus(
            initialized=self._initialized, ml_model_loaded=ml_loaded,
            ml_model_trained_at=None, ml_model_auc=ml_auc,
            rag_memory_count=len(self.rag_memory._cards) if self.rag_memory else 0,
            meta_labeler_loaded=self.meta_labeler is not None,
            monitoring_enabled=self.enable_monitoring,
            total_recommendations=0, recent_win_rate=0.0,
            drift_alerts=[], needs_retrain=False
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Trading System")
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--analyze', type=str, help='Analyze ticker')
    args = parser.parse_args()

    system = AITradingSystem()
    system.initialize()

    if args.status:
        status = system.get_status()
        print(f"\n✅ AI Trading System Status")
        print(f"   Initialized: {status.initialized}")
        print(f"   ML Model: {'Loaded' if status.ml_model_loaded else 'Not loaded'}")
        print(f"   AUC: {status.ml_model_auc:.3f}")

    elif args.analyze:
        scores = {'sentiment_score': 72, 'fundamental_score': 65, 'technical_score': 68,
                  'options_flow_score': 75, 'total_score': 70}
        result = system.analyze(args.analyze, scores)
        print(result.get_summary())