"""
Multi-Factor Alpha Model v2.0 - WITH CRITICAL FIXES
====================================================

A professional-grade quantitative model that learns optimal factor weights
from historical data to predict stock returns.

VERSION 2.0 FIXES APPLIED:
1. SCALER LEAKAGE FIX - fold-wise fit/transform (not global)
2. MISSING DATA FIX - median imputation + missingness indicators (not fillna(50))
3. REGIME FIX - explicit UNKNOWN regime (not silent VIX=20 fallback)
4. IMPORTANCE FIX - permutation importance (not just |weights|)
5. SIGNAL THRESHOLD FIX - quantile-based thresholds (not static)

Location: src/ml/multi_factor_alpha.py

Author: HH Research Platform
Date: January 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from scipy.optimize import minimize

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HGBR_AVAILABLE = True
except ImportError:
    HGBR_AVAILABLE = False

# Database connection
try:
    from src.db.connection import get_engine, get_connection
except ImportError:
    from sqlalchemy import create_engine, text
    from contextlib import contextmanager
    from dotenv import load_dotenv
    load_dotenv()

    _engine = None

    def get_engine():
        global _engine
        if _engine is None:
            host = os.getenv('POSTGRES_HOST', 'localhost')
            port = os.getenv('POSTGRES_PORT', '5432')
            db = os.getenv('POSTGRES_DB', 'alpha_platform')
            user = os.getenv('POSTGRES_USER', 'alpha')
            password = os.getenv('POSTGRES_PASSWORD', '')
            conn_str = f'postgresql://{user}:{password}@{host}:{port}/{db}'
            _engine = create_engine(conn_str, pool_pre_ping=True)
        return _engine

    @contextmanager
    def get_connection():
        engine = get_engine()
        conn = engine.connect()
        try:
            yield conn
        finally:
            conn.close()


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"
    HIGH_VOL = "HIGH_VOL"
    UNKNOWN = "UNKNOWN"  # FIX: Added explicit UNKNOWN regime


class SectorGroup(Enum):
    """Sector groupings for factor adjustments."""
    GROWTH = "GROWTH"
    DEFENSIVE = "DEFENSIVE"
    CYCLICAL = "CYCLICAL"
    OTHER = "OTHER"


@dataclass
class FactorDefinition:
    """Definition of a single factor."""
    name: str
    category: str
    description: str
    expected_sign: int
    decay_days: int
    min_value: float = 0
    max_value: float = 100
    weight_bounds: Tuple[float, float] = (-0.5, 0.5)


@dataclass
class AlphaPrediction:
    """Prediction result from the multi-factor model."""
    ticker: str
    prediction_date: date
    expected_return_5d: float
    expected_return_10d: float
    expected_return_20d: float
    ci_lower_5d: float
    ci_upper_5d: float
    ci_lower_10d: float
    ci_upper_10d: float
    sharpe_ratio_implied: float
    information_ratio: float
    prob_positive_5d: float
    prob_positive_10d: float
    prob_beat_market_5d: float
    factor_contributions: Dict[str, float]
    top_bullish_factors: List[Tuple[str, float]]
    top_bearish_factors: List[Tuple[str, float]]
    prediction_confidence: str
    model_uncertainty: float
    regime: str
    sector: str
    sector_adjustment: float
    signal: str
    conviction: float
    recommended_position_size: float
    feature_coverage: float = 1.0  # FIX: Added coverage tracking

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        return (f"{self.ticker}: {self.signal} | "
                f"E[R_5d]={self.expected_return_5d:+.2%} "
                f"(CI: {self.ci_lower_5d:+.2%} to {self.ci_upper_5d:+.2%}) | "
                f"P(win)={self.prob_positive_5d:.1%} | "
                f"Confidence={self.prediction_confidence}")


@dataclass
class FactorWeights:
    """Learned factor weights for a specific regime/sector combination."""
    regime: MarketRegime
    sector_group: SectorGroup
    horizon_days: int
    weights: Dict[str, float]
    intercept: float
    r_squared: float
    mse: float
    information_coefficient: float
    n_samples: int
    train_start: date
    train_end: date


@dataclass
class ModelValidationReport:
    """Comprehensive validation report."""
    model_version: str
    trained_at: datetime
    overall_ic: float
    overall_icir: float
    overall_r2: float
    regime_performance: Dict[str, Dict[str, float]]
    sector_performance: Dict[str, Dict[str, float]]
    n_folds: int
    fold_results: List[Dict]
    mean_ic_oos: float
    std_ic_oos: float
    factor_importance: Dict[str, float]
    factor_stability: Dict[str, float]
    max_drawdown_from_signals: float
    turnover: float
    ic_tstat: float
    ic_pvalue: float
    beats_baseline: bool

    def print_report(self):
        print("=" * 70)
        print("MULTI-FACTOR ALPHA MODEL VALIDATION REPORT")
        print(f"Version: {self.model_version} | Trained: {self.trained_at}")
        print("=" * 70)
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Information Coefficient (IC): {self.overall_ic:.4f}")
        print(f"   IC Information Ratio (ICIR): {self.overall_icir:.4f}")
        print(f"   R-squared: {self.overall_r2:.4f}")
        print(f"   IC t-stat: {self.ic_tstat:.2f} (p={self.ic_pvalue:.4f})")
        print(f"   Beats baseline: {'✅ YES' if self.beats_baseline else '❌ NO'}")
        print(f"\n📈 WALK-FORWARD VALIDATION ({self.n_folds} folds):")
        print(f"   Mean OOS IC: {self.mean_ic_oos:.4f} ± {self.std_ic_oos:.4f}")
        if self.regime_performance:
            print(f"\n🏭 PERFORMANCE BY REGIME:")
            for regime, metrics in self.regime_performance.items():
                print(f"   {regime}: IC={metrics.get('ic', 0):.4f}, N={metrics.get('n', 0)}")
        if self.factor_importance:
            print(f"\n📊 TOP FACTORS BY IMPORTANCE:")
            sorted_factors = sorted(self.factor_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for factor, importance in sorted_factors:
                stability = self.factor_stability.get(factor, 0)
                print(f"   + {factor}: {importance:.4f} (stability: {stability:.2f})")
        print("=" * 70)


# =============================================================================
# FACTOR DEFINITIONS
# =============================================================================

FACTOR_DEFINITIONS = {
    # =========================================================================
    # CORE FACTORS (good coverage, used in model)
    # =========================================================================
    'fundamental_score': FactorDefinition(
        name='fundamental_score', category='fundamental',
        description='Combined fundamental score (PE, PB, ROE, etc.)',
        expected_sign=1, decay_days=60
    ),
    'growth_score': FactorDefinition(
        name='growth_score', category='fundamental',
        description='Growth factor (revenue growth, earnings growth) - IC +0.011',
        expected_sign=1, decay_days=60
    ),
    'total_score': FactorDefinition(
        name='total_score', category='combined',
        description='Combined platform score',
        expected_sign=1, decay_days=10
    ),
    'gap_score': FactorDefinition(
        name='gap_score', category='technical',
        description='Gap analysis score',
        expected_sign=1, decay_days=3
    ),
    'inst_13f_score': FactorDefinition(
        name='inst_13f_score', category='institutional',
        description='Institutional holdings changes - IC +0.017 (best factor)',
        expected_sign=1, decay_days=90
    ),

    # =========================================================================
    # DERIVED FACTORS
    # =========================================================================
    'value_score': FactorDefinition(
        name='value_score', category='fundamental',
        description='Value composite (derived from fundamental_score)',
        expected_sign=1, decay_days=60
    ),
    'quality_score': FactorDefinition(
        name='quality_score', category='fundamental',
        description='Quality composite (derived from fundamental_score)',
        expected_sign=1, decay_days=60
    ),
    'score_momentum': FactorDefinition(
        name='score_momentum', category='momentum',
        description='Change in total_score (momentum signal)',
        expected_sign=1, decay_days=5, min_value=-50, max_value=50
    ),

    # =========================================================================
    # EXCLUDED FACTORS (documented for reference)
    # These are NOT used due to issues found in diagnostics
    # =========================================================================
    'technical_score': FactorDefinition(
        name='technical_score', category='technical',
        description='EXCLUDED: 21.7% coverage, IC -0.070 (contrarian)',
        expected_sign=1, decay_days=5  # Note: inverted if used
    ),
    'sentiment_score': FactorDefinition(
        name='sentiment_score', category='sentiment',
        description='EXCLUDED: Corrupt data (std=1.5M)',
        expected_sign=1, decay_days=5
    ),
    'options_score': FactorDefinition(
        name='options_score', category='options',
        description='EXCLUDED: 14.2% coverage, IC -0.041 (contrarian)',
        expected_sign=1, decay_days=5
    ),
    'short_squeeze_score': FactorDefinition(
        name='short_squeeze_score', category='options',
        description='EXCLUDED: 14.4% coverage, IC -0.064 (contrarian)',
        expected_sign=1, decay_days=10
    ),
    'insider_score': FactorDefinition(
        name='insider_score', category='institutional',
        description='EXCLUDED: 9.7% coverage, IC -0.078 (strong contrarian)',
        expected_sign=1, decay_days=30
    ),
    'target_upside_pct': FactorDefinition(
        name='target_upside_pct', category='sentiment',
        description='EXCLUDED: 10.5% coverage',
        expected_sign=1, decay_days=60, min_value=-50, max_value=100
    ),
    'analyst_positivity': FactorDefinition(
        name='analyst_positivity', category='sentiment',
        description='EXCLUDED: 6.5% coverage (almost useless)',
        expected_sign=1, decay_days=30
    ),
}

SECTOR_TO_GROUP = {
    'Technology': SectorGroup.GROWTH,
    'Consumer Cyclical': SectorGroup.GROWTH,
    'Communication Services': SectorGroup.GROWTH,
    'Utilities': SectorGroup.DEFENSIVE,
    'Consumer Defensive': SectorGroup.DEFENSIVE,
    'Healthcare': SectorGroup.DEFENSIVE,
    'Financials': SectorGroup.CYCLICAL,
    'Industrials': SectorGroup.CYCLICAL,
    'Materials': SectorGroup.CYCLICAL,
    'Energy': SectorGroup.CYCLICAL,
    'Real Estate': SectorGroup.CYCLICAL,
}


# =============================================================================
# ALPHA DATA LOADER - WITH FIXES
# =============================================================================

class AlphaDataLoader:
    """
    Loads and prepares factor data for the alpha model.

    FIX APPLIED: Proper missing data handling, no more fillna(50)
    PATCH APPLIED: Frozen schema for train/test consistency
    """

    def __init__(self):
        self.engine = get_engine()
        self.scaler = RobustScaler()
        self._factor_stats = {}

        # FIX: Store per-feature statistics for proper imputation
        self._feature_medians = {}
        self._feature_stds = {}
        self._feature_coverage = {}

        # PATCH Step 1A: Frozen schema (base + indicator features) after fitting
        self._base_features: List[str] = []
        self._indicator_features: List[str] = []
        self._frozen_feature_schema: List[str] = []

    def load_historical_data(self,
                             min_date: Optional[str] = None,
                             max_date: Optional[str] = None,
                             min_samples_per_ticker: int = 20) -> pd.DataFrame:
        """Load historical factor data with forward returns."""

        query = """
            WITH signal_data AS (
                SELECT 
                    h.ticker,
                    h.score_date as date,
                    h.sector,
                    h.sentiment as sentiment_score,
                    h.fundamental_score as fundamental_score,
                    h.growth_score as growth_score,
                    h.total_score as total_score,
                    h.gap_score as gap_score,
                    s.technical_score as technical_score,
                    s.options_flow_score as options_score,
                    s.short_squeeze_score as short_squeeze_score,
                    s.target_upside_pct as target_upside_pct,
                    s.analyst_positivity as analyst_positivity,
                    s.insider_signal as insider_score,
                    s.institutional_signal as inst_13f_score,
                    h.op_price as price,
                    h.return_1d,
                    h.return_5d,
                    h.return_10d,
                    h.return_20d
                FROM historical_scores h
                LEFT JOIN screener_scores s 
                    ON h.ticker = s.ticker AND h.score_date = s.date
                WHERE h.op_price IS NOT NULL 
                    AND h.op_price > 0
                    AND h.return_5d IS NOT NULL
            )
            SELECT * FROM signal_data
            WHERE date >= COALESCE(%(min_date)s::date, '2020-01-01')
              AND date <= COALESCE(%(max_date)s::date, CURRENT_DATE - INTERVAL '5 days')
            ORDER BY date, ticker
        """

        params = {'min_date': min_date, 'max_date': max_date}

        try:
            df = pd.read_sql(query, self.engine, params=params)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()

        if df.empty:
            logger.warning("No data loaded from database")
            return df

        logger.info(f"Loaded {len(df)} samples from {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Unique tickers: {df['ticker'].nunique()}")

        # Filter tickers with minimum samples
        ticker_counts = df.groupby('ticker').size()
        valid_tickers = ticker_counts[ticker_counts >= min_samples_per_ticker].index
        df = df[df['ticker'].isin(valid_tickers)]
        logger.info(f"After filtering: {len(df)} samples, {df['ticker'].nunique()} tickers")

        # FIX: Convert text gap_score to numeric
        df = self._convert_gap_score(df)

        # Add derived factors
        df = self._add_derived_factors(df)

        # FIX: Add regime data with proper handling
        df = self._add_regime_data(df)

        # Add sector group
        df['sector_group'] = df['sector'].map(SECTOR_TO_GROUP).fillna(SectorGroup.OTHER)
        df['sector_group'] = df['sector_group'].apply(lambda x: x.value if isinstance(x, SectorGroup) else x)

        return df

    def _convert_gap_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert text-based gap_score to numeric."""
        if 'gap_score' not in df.columns:
            return df

        if df['gap_score'].dtype == 'object':
            gap_score_map = {
                'Stong Up': 90, 'Strong Up': 90,
                'Gap Up Continuation': 80,
                'Potential Up': 70,
                'No Signal Up': 60,
                'No Analysis': 50, 'No Signal': 50, 'Reversal': 50,
                'No Signal Down': 40,
                'Potential Down': 30,
                'Gap Down': 20, 'Gap Down Continuation': 20,
                'Strong Down': 10, 'Stong Down': 10,
            }
            df['gap_score'] = df['gap_score'].map(gap_score_map)
            # Don't fill NA here - let proper imputation handle it

        return df

    def _add_derived_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived/engineered factors.

        DIAGNOSTIC FIXES APPLIED:
        1. Clean corrupt sentiment_score values
        2. Invert contrarian factors (if present)
        3. Cap extreme values
        """

        # =====================================================================
        # FIX: Clean corrupt data
        # =====================================================================

        # sentiment_score had std=1.5M which is impossible - cap to valid range
        if 'sentiment_score' in df.columns:
            # Cap to reasonable range (0-100 scale)
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
            df.loc[df['sentiment_score'] > 100, 'sentiment_score'] = np.nan
            df.loc[df['sentiment_score'] < 0, 'sentiment_score'] = np.nan
            logger.info(f"Cleaned sentiment_score: {df['sentiment_score'].notna().sum()} valid values")

        # Cap all score columns to 0-100 range
        score_cols = ['fundamental_score', 'growth_score', 'total_score',
                      'technical_score', 'options_score', 'short_squeeze_score',
                      'insider_score', 'inst_13f_score', 'analyst_positivity']
        for col in score_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].clip(0, 100)

        # =====================================================================
        # FIX: Invert contrarian factors (negative IC → positive)
        # Based on diagnostics:
        # - technical_score: IC -0.070 → invert
        # - insider_score: IC -0.078 → invert
        # - options_score: IC -0.041 → invert
        # - short_squeeze_score: IC -0.064 → invert
        # =====================================================================

        contrarian_factors = ['technical_score', 'insider_score', 'options_score', 'short_squeeze_score']
        for col in contrarian_factors:
            if col in df.columns:
                # Invert: 100 - score (so high becomes low, low becomes high)
                valid_mask = df[col].notna()
                if valid_mask.any():
                    df.loc[valid_mask, col] = 100 - df.loc[valid_mask, col]
                    logger.debug(f"Inverted contrarian factor: {col}")

        # =====================================================================
        # Derived factors
        # =====================================================================

        # Score momentum (change in total score)
        df = df.sort_values(['ticker', 'date'])
        df['score_momentum'] = df.groupby('ticker')['total_score'].diff(1)

        # Don't use sentiment_momentum since sentiment_score is problematic
        # df['sentiment_momentum'] = df.groupby('ticker')['sentiment_score'].diff(1)

        # Value/Quality composites (based on fundamentals)
        if 'fundamental_score' in df.columns:
            df['value_score'] = df['fundamental_score']
            df['quality_score'] = df['fundamental_score']

        # Institutional composite - only use inst_13f_score (positive IC)
        # Don't include insider_score in composite since it's contrarian
        if 'inst_13f_score' in df.columns:
            df['institutional_composite'] = df['inst_13f_score']
        else:
            df['institutional_composite'] = None

        return df

    def _add_regime_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime with PROPER handling of missing data.

        FIX: No more silent VIX=20 fallback - uses explicit UNKNOWN regime.
        """
        try:
            vix_query = """
                SELECT date, close as vix
                FROM prices
                WHERE ticker = '^VIX' OR ticker = 'VIX'
                ORDER BY date
            """
            vix_df = pd.read_sql(vix_query, self.engine)

            if not vix_df.empty:
                vix_df['date'] = pd.to_datetime(vix_df['date'])
                df['date'] = pd.to_datetime(df['date'])
                df = df.merge(vix_df, on='date', how='left')

                # FIX: Track VIX coverage
                vix_coverage = df['vix'].notna().mean()
                logger.info(f"VIX coverage: {vix_coverage:.1%}")

                if vix_coverage < 0.5:
                    logger.warning(f"Low VIX coverage ({vix_coverage:.1%}) - regime classification may be unreliable")
            else:
                df['vix'] = np.nan
                logger.warning("No VIX data found - regime will be UNKNOWN")

        except Exception as e:
            df['vix'] = np.nan
            logger.warning(f"Error loading VIX: {e}")

        # FIX: Classify regime with explicit UNKNOWN for missing data
        def classify_regime(vix):
            if pd.isna(vix):
                return MarketRegime.UNKNOWN.value  # FIX: Explicit unknown
            if vix < 15:
                return MarketRegime.BULL.value
            elif vix < 20:
                return MarketRegime.NEUTRAL.value
            elif vix < 30:
                return MarketRegime.BEAR.value
            else:
                return MarketRegime.HIGH_VOL.value

        df['regime'] = df['vix'].apply(classify_regime)

        # FIX: Add regime uncertainty flag
        df['regime_uncertain'] = df['regime'] == MarketRegime.UNKNOWN.value

        return df

    def get_factor_columns(self) -> List[str]:
        """
        Return list of factor columns to use.

        DIAGNOSTIC FIXES APPLIED:
        - Removed low-coverage factors (<25% coverage)
        - Removed factors with corrupt data (sentiment_score)
        - Kept only factors with decent coverage OR positive IC

        Coverage & IC Analysis (from diagnostics):
        - fundamental_score: 86.8% coverage, IC ~0 ✓
        - growth_score: 93% coverage, IC +0.011 ✓
        - total_score: 87.5% coverage, IC +0.005 ✓
        - inst_13f_score: 9.7% coverage but IC +0.017 ✓
        - gap_score: derived, keep for now ✓

        REMOVED:
        - sentiment_score: BAD DATA (std=1.5M - corrupt values)
        - technical_score: 21.7% coverage, IC -0.070 (contrarian)
        - options_score: 14.2% coverage, IC -0.041 (contrarian)
        - short_squeeze_score: 14.4% coverage, IC -0.064 (contrarian)
        - insider_score: 9.7% coverage, IC -0.078 (strong contrarian)
        - target_upside_pct: 10.5% coverage
        - analyst_positivity: 6.5% coverage (almost useless)
        """
        return [
            # Core factors with good coverage
            'fundamental_score',    # 86.8% coverage
            'growth_score',         # 93% coverage, positive IC
            'total_score',          # 87.5% coverage
            'gap_score',            # Derived from text

            # Institutional factor (low coverage but positive IC)
            'inst_13f_score',       # 9.7% but IC +0.017

            # Derived factors (computed from above)
            'value_score',          # = fundamental_score
            'quality_score',        # = fundamental_score
            'score_momentum',       # diff of total_score

            # Note: We're NOT including these due to issues identified in diagnostics:
            # - sentiment_score (corrupt data)
            # - technical_score, options_score, short_squeeze_score, insider_score (contrarian + low coverage)
            # - target_upside_pct, analyst_positivity (very low coverage)
        ]

    # =========================================================================
    # PATCH Step 1B: Helper methods for schema-frozen feature handling
    # =========================================================================

    def _compute_indicator_features(self, X: pd.DataFrame, base_features: List[str],
                                    threshold: float = 0.05) -> List[str]:
        """Decide which missingness indicators to include based on TRAIN data only."""
        indicators = []
        for col in base_features:
            if col not in X.columns:
                continue
            missing_rate = X[col].isna().mean()
            if missing_rate > threshold:
                indicators.append(f"{col}_missing")
        return indicators

    def _apply_imputation_and_indicators(self,
                                        X: pd.DataFrame,
                                        base_features: List[str],
                                        indicator_features: List[str],
                                        fit: bool) -> pd.DataFrame:
        """
        Build a feature frame with:
          - base features (median-imputed using TRAIN medians)
          - indicator features (0/1; never median-imputed)
        Ensures all requested columns exist and are ordered deterministically.
        """
        df = X.copy()

        # Ensure base columns exist
        for col in base_features:
            if col not in df.columns:
                df[col] = np.nan

        # FIT: compute and store medians only for base features
        if fit:
            for col in base_features:
                med = pd.to_numeric(df[col], errors="coerce").median()
                if pd.isna(med):
                    med = 0.0  # last resort for truly empty columns
                self._feature_medians[col] = float(med)
                self._feature_coverage[col] = float(1.0 - df[col].isna().mean())

        # Apply base imputation using stored medians
        for col in base_features:
            med = self._feature_medians.get(col, 0.0)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(med)

        # Create indicator columns deterministically from base missingness
        # Important: indicator is 1 if ORIGINAL (pre-impute) was missing.
        # We must compute missingness from original X, not from df after fill.
        for ind in indicator_features:
            base = ind[:-8]  # strip "_missing"
            if base in X.columns:
                missing_mask = pd.to_numeric(X[base], errors="coerce").isna()
            else:
                missing_mask = pd.Series(True, index=df.index)  # base absent => missing
            df[ind] = missing_mask.astype(float)

        # Final schema and ordering
        schema = list(base_features) + list(indicator_features)
        for col in schema:
            if col not in df.columns:
                df[col] = 0.0

        return df[schema]

    # =========================================================================
    # PATCH Step 1C: Schema-frozen prepare_features
    # =========================================================================

    def prepare_features(self, df: pd.DataFrame, fit: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare and normalize feature matrix.

        PATCH Fixes:
          - schema is frozen at fit-time (base + indicators)
          - indicators are computed, not imputed
          - deterministic ordering for train/live
        """
        base = [c for c in self.get_factor_columns() if c in df.columns]
        X_raw = df[base].copy() if base else pd.DataFrame(index=df.index)

        if fit:
            self._base_features = list(base)
            self._indicator_features = self._compute_indicator_features(X_raw, self._base_features, threshold=0.05)
            self._frozen_feature_schema = self._base_features + self._indicator_features

        # Use frozen schema on transform
        base_features = self._base_features if self._base_features else list(base)
        indicator_features = self._indicator_features if self._indicator_features else []

        X_feat = self._apply_imputation_and_indicators(
            X_raw, base_features=base_features, indicator_features=indicator_features, fit=fit
        )

        X_array = X_feat.values.astype(np.float64)
        X_array = np.nan_to_num(X_array, nan=0.0, posinf=100.0, neginf=0.0)

        if fit:
            X_array = self.scaler.fit_transform(X_array)
            self._factor_stats = {'mean': getattr(self.scaler, "center_", None),
                                  'scale': getattr(self.scaler, "scale_", None)}
        else:
            if hasattr(self.scaler, 'center_') and self.scaler.center_ is not None:
                X_array = self.scaler.transform(X_array)

        return X_array, list(X_feat.columns)

    def _handle_missing_data(self, X: pd.DataFrame, feature_names: List[str],
                             fit: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        """
        FIX: Proper missing data handling that preserves signal.

        Uses:
        1. Per-feature median imputation (not constant 50)
        2. Missingness indicator features for features with >5% missing
        3. Coverage tracking
        """
        result = X.copy()
        new_feature_names = list(feature_names)

        for col in feature_names:
            if col not in result.columns:
                continue

            missing_rate = result[col].isna().mean()

            if fit:
                # Store median for this feature
                self._feature_medians[col] = result[col].median()
                self._feature_stds[col] = result[col].std()
                self._feature_coverage[col] = 1 - missing_rate

                logger.debug(f"Feature {col}: coverage={1-missing_rate:.1%}, median={self._feature_medians[col]}")

            # FIX: Add missingness indicator if >5% missing (informative missingness)
            if missing_rate > 0.05:
                indicator_col = f'{col}_missing'
                if indicator_col not in result.columns:
                    result[indicator_col] = X[col].isna().astype(float)
                    if indicator_col not in new_feature_names:
                        new_feature_names.append(indicator_col)

            # FIX: Use median imputation (not constant 50!)
            median_val = self._feature_medians.get(col)
            if median_val is None or pd.isna(median_val):
                # Only use 50 as absolute last resort
                median_val = 50.0

            result[col] = result[col].fillna(median_val)

        return result, new_feature_names

    # =========================================================================
    # PATCH Step 1D: Schema-consistent fold feature preparation
    # =========================================================================

    def prepare_features_for_fold(self,
                                  df_train: pd.DataFrame,
                                  df_test: pd.DataFrame,
                                  base_features: List[str],
                                  indicator_threshold: float = 0.05
                                  ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fold-wise feature prep with PATCH fixes:
          - indicators decided from TRAIN only
          - medians computed from TRAIN only
          - identical column set/order for train/test
          - fold scaler fit on TRAIN only

        FIX: Ensure EXACTLY the same columns in both train and test
        """
        fold_scaler = RobustScaler()

        # Use only features that exist in TRAIN (test may have different coverage)
        base = [c for c in base_features if c in df_train.columns]

        # Extract raw data - only use columns from base list
        X_train_raw = df_train[base].copy() if base else pd.DataFrame(index=df_train.index)

        # For test, use the SAME base columns (fill missing columns with NaN)
        X_test_raw = pd.DataFrame(index=df_test.index)
        for col in base:
            if col in df_test.columns:
                X_test_raw[col] = df_test[col]
            else:
                X_test_raw[col] = np.nan

        # Decide indicators from TRAIN only - these are FIXED for both train and test
        indicators = self._compute_indicator_features(X_train_raw, base, threshold=indicator_threshold)

        # Compute medians from TRAIN only
        fold_medians = {}
        for col in base:
            med = pd.to_numeric(X_train_raw[col], errors="coerce").median()
            if pd.isna(med):
                med = 0.0
            fold_medians[col] = float(med)

        # Define the FROZEN schema - same for both train and test
        frozen_schema = list(base) + list(indicators)

        def build_fold_frame(Xraw: pd.DataFrame, original_raw: pd.DataFrame) -> pd.DataFrame:
            """
            Build feature frame with EXACTLY the frozen schema.

            Args:
                Xraw: The raw data (already filtered to base columns)
                original_raw: Original raw data for computing missingness indicators
            """
            df = pd.DataFrame(index=Xraw.index)

            # 1. Add all base features (impute with fold medians)
            for col in base:
                if col in Xraw.columns:
                    df[col] = pd.to_numeric(Xraw[col], errors="coerce").fillna(fold_medians.get(col, 0.0))
                else:
                    df[col] = fold_medians.get(col, 0.0)

            # 2. Add all indicator features (compute from ORIGINAL, pre-imputed data)
            for ind in indicators:
                base_col = ind[:-8]  # strip "_missing"
                if base_col in original_raw.columns:
                    # Missingness from original data
                    mm = pd.to_numeric(original_raw[base_col], errors="coerce").isna()
                else:
                    # Column missing entirely = all missing
                    mm = pd.Series(True, index=df.index)
                df[ind] = mm.astype(float)

            # 3. Ensure exact schema ordering
            for col in frozen_schema:
                if col not in df.columns:
                    df[col] = 0.0

            return df[frozen_schema]

        # Build frames with IDENTICAL schemas
        X_train_feat = build_fold_frame(X_train_raw, X_train_raw)
        X_test_feat = build_fold_frame(X_test_raw, X_test_raw)

        # Verify shapes match
        assert X_train_feat.shape[1] == X_test_feat.shape[1], \
            f"Schema mismatch: train has {X_train_feat.shape[1]} cols, test has {X_test_feat.shape[1]} cols"
        assert list(X_train_feat.columns) == list(X_test_feat.columns), \
            f"Column order mismatch between train and test"

        X_train_array = np.nan_to_num(X_train_feat.values.astype(np.float64), nan=0.0)
        X_test_array = np.nan_to_num(X_test_feat.values.astype(np.float64), nan=0.0)

        X_train_scaled = fold_scaler.fit_transform(X_train_array)
        X_test_scaled = fold_scaler.transform(X_test_array)

        return X_train_scaled, X_test_scaled, list(X_train_feat.columns)

    def load_live_data(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load current factor values for live prediction.

        FIX Step 1:
        - screener_scores.sector may not exist (DB schema issue).
        - Use fundamentals.sector when available.
        - If neither exists, return sector as NULL and still predict using global models.
        """
        query = """
            SELECT DISTINCT ON (s.ticker)
                s.ticker,
                s.date,

                -- sector source (prefer fundamentals table)
                f.sector AS sector,

                s.total_score,
                s.sentiment_score,
                s.fundamental_score,
                s.growth_score,
                s.technical_score,
                s.options_flow_score as options_score,
                s.short_squeeze_score,
                s.target_upside_pct,
                s.analyst_positivity,
                s.insider_signal as insider_score,
                s.institutional_signal as inst_13f_score
            FROM screener_scores s
            LEFT JOIN fundamentals f ON f.ticker = s.ticker
            WHERE s.date >= CURRENT_DATE - INTERVAL '7 days'
        """

        if tickers:
            placeholders = ', '.join([f"'{t}'" for t in tickers])
            query += f" AND s.ticker IN ({placeholders})"

        query += " ORDER BY s.ticker, s.date DESC"

        try:
            df = pd.read_sql(query, self.engine)
        except Exception as e:
            logger.error(f"Error loading live data: {e}")
            return pd.DataFrame()

        if df.empty:
            logger.warning("No live data available")
            return df

        # Ensure sector exists even if fundamentals missing
        if 'sector' not in df.columns:
            df['sector'] = None

        # =====================================================================
        # DIAGNOSTIC FIX: Clean data same as in training
        # =====================================================================

        # Cap all score columns to 0-100 range
        score_cols = ['fundamental_score', 'growth_score', 'total_score',
                      'technical_score', 'options_score', 'short_squeeze_score',
                      'insider_score', 'inst_13f_score', 'analyst_positivity']
        for col in score_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].clip(0, 100)

        # Clean corrupt sentiment_score
        if 'sentiment_score' in df.columns:
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
            df.loc[df['sentiment_score'] > 100, 'sentiment_score'] = np.nan
            df.loc[df['sentiment_score'] < 0, 'sentiment_score'] = np.nan

        # Invert contrarian factors (same as training)
        contrarian_factors = ['technical_score', 'insider_score', 'options_score', 'short_squeeze_score']
        for col in contrarian_factors:
            if col in df.columns:
                valid_mask = df[col].notna()
                if valid_mask.any():
                    df.loc[valid_mask, col] = 100 - df.loc[valid_mask, col]

        # =====================================================================
        # Add derived factors (consistent with training)
        # =====================================================================

        df['score_momentum'] = 0  # Can't compute momentum for single snapshot
        df['value_score'] = df['fundamental_score'] if 'fundamental_score' in df.columns else 50
        df['quality_score'] = df['fundamental_score'] if 'fundamental_score' in df.columns else 50

        # Institutional composite - only use inst_13f_score (positive IC)
        if 'inst_13f_score' in df.columns:
            df['institutional_composite'] = df['inst_13f_score']
        else:
            df['institutional_composite'] = None

        df['gap_score'] = 50  # Default for live

        # Add regime
        df = self._add_regime_data(df)

        # Add sector group
        df['sector_group'] = df['sector'].map(SECTOR_TO_GROUP).fillna(SectorGroup.OTHER)
        df['sector_group'] = df['sector_group'].apply(lambda x: x.value if isinstance(x, SectorGroup) else x)

        return df


# =============================================================================
# MULTI-FACTOR ALPHA MODEL - WITH FIXES
# =============================================================================

class MultiFactorAlphaModel:
    """
    Professional multi-factor alpha model with CRITICAL FIXES:

    1. Fold-wise fit/transform (no scaler leakage)
    2. Proper missing data handling
    3. Permutation importance
    4. Quantile-based signal thresholds
    """

    VERSION = "2.2.1"  # Diagnostic fixes + regime-aware signal adjustment

    def __init__(self,
                 target_horizons: List[int] = [5, 10, 20],
                 min_samples_for_training: int = 500,
                 regularization_alpha: float = 0.1,
                 use_regime_models: bool = True,
                 use_sector_models: bool = True):

        self.target_horizons = target_horizons
        self.min_samples = min_samples_for_training
        self.alpha = regularization_alpha
        self.use_regime_models = use_regime_models
        self.use_sector_models = use_sector_models

        self.data_loader = AlphaDataLoader()

        self.models: Dict[Tuple, Any] = {}
        self.factor_weights: Dict[Tuple, FactorWeights] = {}
        self.global_models: Dict[int, Any] = {}
        self.global_weights: Dict[int, FactorWeights] = {}
        self.feature_names: List[str] = []
        self.validation_report: Optional[ModelValidationReport] = None
        self.return_stats: Dict[int, Dict[str, float]] = {}
        self.factor_importance: Dict[str, float] = {}

        # FIX: Store permutation importance
        self._permutation_importance: Dict[str, Dict] = {}

        self._is_trained = False

    def train(self,
              min_date: Optional[str] = None,
              max_date: Optional[str] = None,
              n_folds: int = 5,
              purge_days: int = 5) -> ModelValidationReport:
        """
        Train the model with walk-forward validation.

        FIX: Uses fold-wise fit/transform to avoid data leakage.
        """
        logger.info("Starting multi-factor alpha model training (v2.0 with fixes)...")

        # Load data
        df = self.data_loader.load_historical_data(min_date, max_date)

        if len(df) < self.min_samples:
            raise ValueError(f"Insufficient data: {len(df)} samples, need {self.min_samples}")

        # Get feature names (but DON'T fit scaler globally!)
        self.feature_names = self.data_loader.get_factor_columns()
        available_features = [c for c in self.feature_names if c in df.columns]
        self.feature_names = available_features

        logger.info(f"Training with {len(df)} samples, {len(self.feature_names)} features")
        logger.info(f"Features: {self.feature_names}")

        # Store return statistics for confidence intervals
        for horizon in self.target_horizons:
            returns = df[f'return_{horizon}d'].dropna()
            self.return_stats[horizon] = {
                'mean': returns.mean(),
                'std': returns.std(),
                'median': returns.median(),
                'q25': returns.quantile(0.25),
                'q75': returns.quantile(0.75)
            }

        # FIX: Walk-forward validation with fold-wise fit/transform
        fold_results = self._walk_forward_validation_fixed(df, n_folds, purge_days)

        # Train final models on all data
        X, self.feature_names = self.data_loader.prepare_features(df, fit=True)
        self._train_final_models(df, X)

        # Generate validation report
        self.validation_report = self._generate_validation_report(df, fold_results)

        self._is_trained = True

        logger.info("Training complete!")
        self.validation_report.print_report()

        return self.validation_report

    def _walk_forward_validation_fixed(self,
                                        df: pd.DataFrame,
                                        n_folds: int,
                                        purge_days: int) -> List[Dict]:
        """
        FIX: Walk-forward validation with FOLD-WISE fit/transform.

        Each fold fits its own scaler on training data only.
        """
        results = []
        all_permutation_importance = []

        dates = np.sort(df['date'].unique())
        n_dates = len(dates)
        test_size = n_dates // (n_folds + 1)

        for fold in range(n_folds):
            # Define train/test periods
            test_end_idx = n_dates - (fold * test_size)
            test_start_idx = test_end_idx - test_size
            train_end_idx = test_start_idx - purge_days

            if train_end_idx <= n_dates // 4:
                continue

            train_dates = dates[:train_end_idx]
            test_dates = dates[test_start_idx:test_end_idx]

            train_mask = df['date'].isin(train_dates)
            test_mask = df['date'].isin(test_dates)

            df_train = df[train_mask].copy()
            df_test = df[test_mask].copy()

            # FIX: Fold-wise feature preparation
            X_train, X_test, feature_names = self.data_loader.prepare_features_for_fold(
                df_train, df_test, self.feature_names
            )

            # PATCH Step 7: Assertions to prevent silent failures
            if X_train.shape[1] != X_test.shape[1]:
                raise ValueError(f"Fold {fold}: train/test feature mismatch {X_train.shape[1]} vs {X_test.shape[1]}")
            if len(feature_names) != X_train.shape[1]:
                raise ValueError(f"Fold {fold}: feature_names length mismatch {len(feature_names)} vs {X_train.shape[1]}")

            fold_result = {'fold': fold, 'train_size': len(X_train), 'test_size': len(X_test)}

            for horizon in self.target_horizons:
                y_col = f'return_{horizon}d'
                y_train = df_train[y_col].values
                y_test = df_test[y_col].values

                # Remove NaN
                valid_train = ~np.isnan(y_train)
                valid_test = ~np.isnan(y_test)

                if valid_train.sum() < 50 or valid_test.sum() < 10:
                    continue

                # Train model on this fold's training data
                model = Ridge(alpha=self.alpha)
                model.fit(X_train[valid_train], y_train[valid_train])

                # Predict on test (true out-of-sample)
                y_pred = model.predict(X_test[valid_test])
                y_actual = y_test[valid_test]

                # Calculate IC
                ic = np.corrcoef(y_pred, y_actual)[0, 1] if len(y_pred) > 2 else 0

                # Calculate other metrics
                mse = mean_squared_error(y_actual, y_pred)
                r2 = r2_score(y_actual, y_pred) if len(y_actual) > 2 else 0

                fold_result[f'ic_{horizon}d'] = ic
                fold_result[f'mse_{horizon}d'] = mse
                fold_result[f'r2_{horizon}d'] = r2

                # FIX: Calculate permutation importance on OOS data
                if horizon == 5:  # Primary horizon
                    perm_imp = self._calculate_permutation_importance(
                        model, X_test[valid_test], y_actual, feature_names
                    )
                    all_permutation_importance.append(perm_imp)

            results.append(fold_result)
            logger.info(f"Fold {fold}: Train={len(X_train)}, Test={len(X_test)}, "
                       f"IC_5d={fold_result.get('ic_5d', 0):.4f}")

        # FIX: Aggregate permutation importance across folds
        self._aggregate_permutation_importance(all_permutation_importance)

        return results

    def _calculate_permutation_importance(self, model, X_test, y_test,
                                          feature_names, n_repeats=3) -> Dict[str, float]:
        """
        FIX: Calculate TRUE out-of-sample importance via permutation.

        This measures how much IC drops when each feature is shuffled.
        """
        # PATCH Step 5: NaN-safe IC calculation
        def safe_ic(a, b) -> float:
            if len(a) < 3:
                return 0.0
            ic = np.corrcoef(a, b)[0, 1]
            if np.isnan(ic) or np.isinf(ic):
                return 0.0
            return float(ic)

        # Baseline IC
        y_pred_base = model.predict(X_test)
        ic_base = safe_ic(y_pred_base, y_test)

        importance = {}

        for i, feature in enumerate(feature_names):
            if i >= X_test.shape[1]:
                continue

            ic_drops = []

            for _ in range(n_repeats):
                # Copy and shuffle one feature
                X_permuted = X_test.copy()
                np.random.shuffle(X_permuted[:, i])

                # Predict with shuffled feature
                y_pred_perm = model.predict(X_permuted)
                ic_perm = safe_ic(y_pred_perm, y_test)

                # IC drop = importance
                ic_drops.append(ic_base - ic_perm)

            importance[feature] = max(0, np.mean(ic_drops))

        return importance

    def _aggregate_permutation_importance(self, all_importance: List[Dict[str, float]]):
        """Aggregate permutation importance across folds."""
        if not all_importance:
            return

        # Collect all features
        all_features = set()
        for imp in all_importance:
            all_features.update(imp.keys())

        # Average importance across folds
        aggregated = {}
        for feature in all_features:
            values = [imp.get(feature, 0) for imp in all_importance]
            aggregated[feature] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'stability': 1 - (np.std(values) / (np.mean(values) + 1e-6))  # Higher = more stable
            }

        self._permutation_importance = aggregated

        # Update factor_importance with permutation-based values
        total_imp = sum(v['mean'] for v in aggregated.values()) + 1e-6
        for feature, stats in aggregated.items():
            self.factor_importance[feature] = stats['mean'] / total_imp

    def _train_final_models(self, df: pd.DataFrame, X: np.ndarray):
        """Train final models on all data."""

        # 1. Train global models
        for horizon in self.target_horizons:
            y_col = f'return_{horizon}d'
            y = df[y_col].values
            valid = ~np.isnan(y)

            if valid.sum() < 100:
                continue

            model = Ridge(alpha=self.alpha)
            model.fit(X[valid], y[valid])

            self.global_models[horizon] = model

            weights_dict = dict(zip(self.feature_names, model.coef_))
            ic = np.corrcoef(model.predict(X[valid]), y[valid])[0, 1]

            self.global_weights[horizon] = FactorWeights(
                regime=MarketRegime.NEUTRAL,
                sector_group=SectorGroup.OTHER,
                horizon_days=horizon,
                weights=weights_dict,
                intercept=model.intercept_,
                r_squared=r2_score(y[valid], model.predict(X[valid])),
                mse=mean_squared_error(y[valid], model.predict(X[valid])),
                information_coefficient=ic,
                n_samples=valid.sum(),
                train_start=df['date'].min(),
                train_end=df['date'].max()
            )

            # Update factor importance (as fallback if permutation not available)
            if not self.factor_importance:
                for factor, weight in weights_dict.items():
                    if factor not in self.factor_importance:
                        self.factor_importance[factor] = 0
                    self.factor_importance[factor] += abs(weight)

        # 2. Train regime-specific models
        if self.use_regime_models:
            for regime in df['regime'].unique():
                if regime == MarketRegime.UNKNOWN.value:
                    continue  # FIX: Don't train on unknown regime

                regime_mask = df['regime'] == regime
                if regime_mask.sum() < 200:
                    continue

                for horizon in self.target_horizons:
                    y_col = f'return_{horizon}d'
                    y = df.loc[regime_mask, y_col].values
                    X_regime = X[regime_mask]
                    valid = ~np.isnan(y)

                    if valid.sum() < 100:
                        continue

                    model = Ridge(alpha=self.alpha)
                    model.fit(X_regime[valid], y[valid])

                    key = (regime, 'ALL', horizon)
                    self.models[key] = model

                    weights_dict = dict(zip(self.feature_names, model.coef_))
                    try:
                        regime_enum = MarketRegime(regime)
                    except:
                        regime_enum = MarketRegime.NEUTRAL

                    self.factor_weights[key] = FactorWeights(
                        regime=regime_enum,
                        sector_group=SectorGroup.OTHER,
                        horizon_days=horizon,
                        weights=weights_dict,
                        intercept=model.intercept_,
                        r_squared=r2_score(y[valid], model.predict(X_regime[valid])),
                        mse=mean_squared_error(y[valid], model.predict(X_regime[valid])),
                        information_coefficient=np.corrcoef(model.predict(X_regime[valid]), y[valid])[0, 1],
                        n_samples=valid.sum(),
                        train_start=df.loc[regime_mask, 'date'].min(),
                        train_end=df.loc[regime_mask, 'date'].max()
                    )

        # 3. Train sector-specific models
        if self.use_sector_models:
            for sector_group in df['sector_group'].unique():
                sector_mask = df['sector_group'] == sector_group
                if sector_mask.sum() < 200:
                    continue

                for horizon in self.target_horizons:
                    y_col = f'return_{horizon}d'
                    y = df.loc[sector_mask, y_col].values
                    X_sector = X[sector_mask]
                    valid = ~np.isnan(y)

                    if valid.sum() < 100:
                        continue

                    model = Ridge(alpha=self.alpha)
                    model.fit(X_sector[valid], y[valid])

                    key = ('ALL', sector_group, horizon)
                    self.models[key] = model

                    weights_dict = dict(zip(self.feature_names, model.coef_))
                    self.factor_weights[key] = FactorWeights(
                        regime=MarketRegime.NEUTRAL,
                        sector_group=SectorGroup(sector_group) if sector_group in [s.value for s in SectorGroup] else SectorGroup.OTHER,
                        horizon_days=horizon,
                        weights=weights_dict,
                        intercept=model.intercept_,
                        r_squared=r2_score(y[valid], model.predict(X_sector[valid])),
                        mse=mean_squared_error(y[valid], model.predict(X_sector[valid])),
                        information_coefficient=np.corrcoef(model.predict(X_sector[valid]), y[valid])[0, 1],
                        n_samples=valid.sum(),
                        train_start=df.loc[sector_mask, 'date'].min(),
                        train_end=df.loc[sector_mask, 'date'].max()
                    )

        # Normalize factor importance
        total_importance = sum(self.factor_importance.values()) or 1
        self.factor_importance = {k: v / total_importance for k, v in self.factor_importance.items()}

        logger.info(f"Trained {len(self.global_models)} global models, "
                   f"{len(self.models)} conditional models")

    def _generate_validation_report(self, df: pd.DataFrame,
                                     fold_results: List[Dict]) -> ModelValidationReport:
        """Generate validation report."""

        # Calculate overall metrics from fold results
        ics_5d = [r.get('ic_5d', 0) for r in fold_results if 'ic_5d' in r]
        mean_ic = np.mean(ics_5d) if ics_5d else 0
        std_ic = np.std(ics_5d) if ics_5d else 1
        icir = mean_ic / std_ic if std_ic > 0 else 0

        # Statistical significance
        if len(ics_5d) > 1:
            t_stat, p_value = stats.ttest_1samp(ics_5d, 0)
        else:
            t_stat, p_value = 0, 1

        # Regime performance
        regime_performance = {}
        for regime in df['regime'].unique():
            if regime == MarketRegime.UNKNOWN.value:
                continue
            regime_df = df[df['regime'] == regime]
            if len(regime_df) < 50:
                continue

            X_regime, _ = self.data_loader.prepare_features(regime_df, fit=False)
            y_regime = regime_df['return_5d'].values
            valid = ~np.isnan(y_regime)

            if valid.sum() < 20 or 5 not in self.global_models:
                continue

            y_pred = self.global_models[5].predict(X_regime[valid])
            ic = np.corrcoef(y_pred, y_regime[valid])[0, 1] if len(y_pred) > 2 else 0
            r2 = r2_score(y_regime[valid], y_pred) if len(y_pred) > 2 else 0

            regime_performance[regime] = {'ic': ic, 'r2': r2, 'n': valid.sum()}

        # Sector performance
        sector_performance = {}
        for sector in df['sector_group'].unique():
            sector_df = df[df['sector_group'] == sector]
            if len(sector_df) < 50:
                continue

            X_sector, _ = self.data_loader.prepare_features(sector_df, fit=False)
            y_sector = sector_df['return_5d'].values
            valid = ~np.isnan(y_sector)

            if valid.sum() < 20 or 5 not in self.global_models:
                continue

            y_pred = self.global_models[5].predict(X_sector[valid])
            ic = np.corrcoef(y_pred, y_sector[valid])[0, 1] if len(y_pred) > 2 else 0
            r2 = r2_score(y_sector[valid], y_pred) if len(y_pred) > 2 else 0

            sector_performance[sector] = {'ic': ic, 'r2': r2, 'n': valid.sum()}

        # FIX: Use actual permutation importance stability
        factor_stability = {}
        for factor in self.factor_importance.keys():
            if factor in self._permutation_importance:
                factor_stability[factor] = self._permutation_importance[factor].get('stability', 0.5)
            else:
                factor_stability[factor] = 0.5  # Unknown stability

        # Overall R2
        if 5 in self.global_models:
            X_all, _ = self.data_loader.prepare_features(df, fit=False)
            y_all = df['return_5d'].values
            valid = ~np.isnan(y_all)
            y_pred_all = self.global_models[5].predict(X_all[valid])
            overall_r2 = r2_score(y_all[valid], y_pred_all)
        else:
            overall_r2 = 0

        return ModelValidationReport(
            model_version=self.VERSION,
            trained_at=datetime.now(),
            overall_ic=mean_ic,
            overall_icir=icir,
            overall_r2=overall_r2,
            regime_performance=regime_performance,
            sector_performance=sector_performance,
            n_folds=len(fold_results),
            fold_results=fold_results,
            mean_ic_oos=mean_ic,
            std_ic_oos=std_ic,
            factor_importance=self.factor_importance,
            factor_stability=factor_stability,
            max_drawdown_from_signals=0,
            turnover=0,
            ic_tstat=t_stat,
            ic_pvalue=p_value,
            beats_baseline=mean_ic > 0.02 and p_value < 0.1
        )

    def predict(self,
                ticker: str,
                factor_values: Dict[str, float],
                sector: str = None,
                regime: str = None) -> AlphaPrediction:
        """Generate alpha prediction for a single stock."""

        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # FIX: Check feature coverage (PATCH Step 4 applied in _check_feature_coverage)
        coverage, missing_features = self._check_feature_coverage(factor_values)

        # PATCH Step 2: Prepare feature vector with correct indicator handling
        feature_vector = []
        for f in self.feature_names:
            # PATCH: Indicators must be computed, never imputed
            if f.endswith("_missing"):
                base = f[:-8]
                base_val = factor_values.get(base)
                is_missing = (base_val is None) or (isinstance(base_val, float) and np.isnan(base_val))
                feature_vector.append(1.0 if is_missing else 0.0)
                continue

            val = factor_values.get(f)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                # Use stored median for base features; default to 0.0 if unknown
                val = self.data_loader._feature_medians.get(f, 0.0)
            feature_vector.append(float(val))

        X = np.array([feature_vector])
        X = self.data_loader.scaler.transform(X)

        # Determine sector group
        sector_group = SECTOR_TO_GROUP.get(sector, SectorGroup.OTHER)
        if isinstance(sector_group, SectorGroup):
            sector_group = sector_group.value

        if regime is None:
            regime = MarketRegime.NEUTRAL.value

        # Get predictions from different models and ensemble
        predictions = {}
        factor_contributions_all = {}

        for horizon in self.target_horizons:
            preds = []
            weights_used = []

            # 1. Global model
            if horizon in self.global_models:
                pred = self.global_models[horizon].predict(X)[0]
                preds.append(pred)
                weights_used.append(self.global_weights[horizon])

            # 2. Regime-specific model
            regime_key = (regime, 'ALL', horizon)
            if regime_key in self.models and regime_key in self.factor_weights:
                pred = self.models[regime_key].predict(X)[0]
                preds.append(pred)
                weights_used.append(self.factor_weights[regime_key])

            # 3. Sector-specific model
            sector_key = ('ALL', sector_group, horizon)
            if sector_key in self.models and sector_key in self.factor_weights:
                pred = self.models[sector_key].predict(X)[0]
                preds.append(pred)
                weights_used.append(self.factor_weights[sector_key])

            # Ensemble
            if preds:
                ic_weights = [max(w.information_coefficient, 0.01) for w in weights_used]
                total_ic = sum(ic_weights)
                ensemble_pred = sum(p * ic / total_ic for p, ic in zip(preds, ic_weights))
                predictions[horizon] = ensemble_pred

                if weights_used:
                    primary_weights = weights_used[0].weights
                    factor_contributions_all[horizon] = {
                        f: primary_weights.get(f, 0) * X[0, i]
                        for i, f in enumerate(self.feature_names) if i < X.shape[1]
                    }

        # Calculate confidence intervals
        def get_ci(horizon, pred):
            stats_h = self.return_stats.get(horizon, {'std': 5})
            std = stats_h['std']
            model_std = std * 1.2
            ci_lower = pred - 1.96 * model_std
            ci_upper = pred + 1.96 * model_std
            return ci_lower, ci_upper

        # Get factor contributions
        factor_contribs = factor_contributions_all.get(5, {})

        # Sort into bullish and bearish
        sorted_contribs = sorted(factor_contribs.items(), key=lambda x: x[1], reverse=True)
        top_bullish = [(f, c) for f, c in sorted_contribs if c > 0][:5]
        top_bearish = [(f, c) for f, c in sorted_contribs if c < 0][:5]

        # Calculate probabilities
        pred_5d = predictions.get(5, 0)
        pred_10d = predictions.get(10, 0)
        stats_5d = self.return_stats.get(5, {'std': 5, 'mean': 0})
        stats_10d = self.return_stats.get(10, {'std': 7, 'mean': 0})

        prob_pos_5d = 1 - stats.norm.cdf(0, loc=pred_5d, scale=stats_5d['std'])
        prob_pos_10d = 1 - stats.norm.cdf(0, loc=pred_10d, scale=stats_10d['std'])

        market_5d = 0.25
        prob_beat_5d = 1 - stats.norm.cdf(market_5d, loc=pred_5d, scale=stats_5d['std'])

        # Sharpe and IR
        sharpe_5d = (pred_5d / stats_5d['std'] * np.sqrt(252/5)) if stats_5d['std'] > 0 else 0
        ir = pred_5d / stats_5d['std'] if stats_5d['std'] > 0 else 0

        # Uncertainty
        if len(predictions) >= 2:
            uncertainty = np.std(list(predictions.values())[:3])
        else:
            uncertainty = stats_5d['std']

        # FIX: Generate signal (will be updated in predict_live with quantile-based)
        signal, conviction, confidence = self._generate_signal(pred_5d, prob_pos_5d, coverage)

        # Position sizing
        base_size = 1.0
        size_adj = conviction * (1 - uncertainty / (2 * stats_5d['std']))
        # FIX: Reduce size if low coverage
        if coverage < 0.8:
            size_adj *= coverage
        recommended_size = max(0.25, min(2.0, base_size * size_adj))

        ci_5d = get_ci(5, pred_5d)
        ci_10d = get_ci(10, predictions.get(10, pred_5d * 1.5))

        return AlphaPrediction(
            ticker=ticker,
            prediction_date=date.today(),
            expected_return_5d=pred_5d / 100,
            expected_return_10d=predictions.get(10, pred_5d * 1.5) / 100,
            expected_return_20d=predictions.get(20, pred_5d * 2.5) / 100,
            ci_lower_5d=ci_5d[0] / 100,
            ci_upper_5d=ci_5d[1] / 100,
            ci_lower_10d=ci_10d[0] / 100,
            ci_upper_10d=ci_10d[1] / 100,
            sharpe_ratio_implied=sharpe_5d,
            information_ratio=ir,
            prob_positive_5d=prob_pos_5d,
            prob_positive_10d=prob_pos_10d,
            prob_beat_market_5d=prob_beat_5d,
            factor_contributions=factor_contribs,
            top_bullish_factors=top_bullish,
            top_bearish_factors=top_bearish,
            prediction_confidence=confidence,
            model_uncertainty=uncertainty,
            regime=regime,
            sector=sector or "Unknown",
            sector_adjustment=0,
            signal=signal,
            conviction=conviction,
            recommended_position_size=recommended_size,
            feature_coverage=coverage
        )

    def _check_feature_coverage(self, factor_values: Dict) -> Tuple[float, List[str]]:
        """Check feature coverage for a prediction.

        PATCH Step 4: Ignores indicator features - coverage should reflect
        real factor availability, not derived indicator fields.
        """
        missing = []
        total_importance = 0
        missing_importance = 0

        for feature, importance in self.factor_importance.items():
            # PATCH: Skip indicator features
            if feature.endswith("_missing"):
                continue
            total_importance += importance
            value = factor_values.get(feature)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                missing.append(feature)
                missing_importance += importance

        coverage = 1 - (missing_importance / total_importance) if total_importance > 0 else 1.0
        return coverage, missing

    def _generate_signal(self, pred_5d: float, prob_pos_5d: float,
                         coverage: float = 1.0) -> Tuple[str, float, str]:
        """
        Generate signal using relaxed thresholds.

        Note: predict_live() will override with quantile-based thresholds.
        """
        # FIX: More relaxed thresholds
        if pred_5d > 1.0 and prob_pos_5d > 0.55:
            signal = "STRONG_BUY"
            conviction = min(0.6 + (prob_pos_5d - 0.5) * 0.6, 0.95)
        elif pred_5d > 0.3 and prob_pos_5d > 0.52:
            signal = "BUY"
            conviction = 0.55 + (prob_pos_5d - 0.5) * 0.5
        elif pred_5d < -1.0 and prob_pos_5d < 0.45:
            signal = "STRONG_SELL"
            conviction = min(0.6 + (0.5 - prob_pos_5d) * 0.6, 0.95)
        elif pred_5d < -0.3 and prob_pos_5d < 0.48:
            signal = "SELL"
            conviction = 0.55 + (0.5 - prob_pos_5d) * 0.5
        else:
            signal = "HOLD"
            conviction = 0.5

        # FIX: Downgrade if low coverage
        if coverage < 0.6:
            conviction = max(0.4, conviction - 0.2)

        # Confidence
        distance_from_50 = abs(prob_pos_5d - 0.5)
        if distance_from_50 > 0.08 and abs(pred_5d) > 0.5 and coverage > 0.7:
            confidence = "HIGH"
        elif distance_from_50 > 0.03 or abs(pred_5d) > 0.2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # FIX: Low coverage = low confidence
        if coverage < 0.6:
            confidence = "LOW"

        return signal, conviction, confidence

    def predict_live(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate live predictions with QUANTILE-BASED signal thresholds.

        FIX: Signals are assigned based on prediction distribution, not static thresholds.
        PATCH Step 3: Build factor_values consistently for indicators.
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Load live data
        df = self.data_loader.load_live_data(tickers)

        if df.empty:
            logger.warning("No live data available")
            return pd.DataFrame()

        # PHASE 1: Generate raw predictions for all stocks
        raw_predictions = []

        for _, row in df.iterrows():
            # PATCH Step 3: Build factor_values correctly for indicators
            factor_values = {}
            for f in self.feature_names:
                if f.endswith("_missing"):
                    base = f[:-8]
                    base_val = row.get(base, None)
                    is_missing = (base_val is None) or (isinstance(base_val, float) and np.isnan(base_val))
                    factor_values[f] = 1.0 if is_missing else 0.0
                else:
                    factor_values[f] = row.get(f, None)

            try:
                pred = self.predict(
                    ticker=row.get('ticker', 'UNKNOWN'),
                    factor_values=factor_values,
                    sector=row.get('sector'),
                    regime=row.get('regime')
                )
                raw_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Error predicting {row.get('ticker')}: {e}")
                continue

        if not raw_predictions:
            return pd.DataFrame()

        # PHASE 2: FIX - Quantile-based signal assignment
        all_pred_5d = [p.expected_return_5d * 100 for p in raw_predictions]

        # Calculate quantiles
        if len(all_pred_5d) > 10:
            q95 = np.percentile(all_pred_5d, 95)
            q85 = np.percentile(all_pred_5d, 85)
            q15 = np.percentile(all_pred_5d, 15)
            q05 = np.percentile(all_pred_5d, 5)
        else:
            # Fallback to fixed thresholds
            q95, q85, q15, q05 = 1.0, 0.5, -0.5, -1.0

        # PHASE 3: Assign signals based on quantiles with REGIME AWARENESS
        # Model performance by regime (from diagnostics):
        # - NEUTRAL: IC = 0.1429 ✓ (good)
        # - HIGH_VOL: IC = 0.1819 ✓ (excellent)
        # - BULL: IC = -0.0641 ✗ (contrarian/poor)
        # - BEAR: IC = -0.0417 ✗ (contrarian/poor)

        FAVORABLE_REGIMES = ['NEUTRAL', 'HIGH_VOL']
        UNFAVORABLE_REGIMES = ['BULL', 'BEAR']

        results = []
        for pred in raw_predictions:
            pred_5d = pred.expected_return_5d * 100
            prob_pos = pred.prob_positive_5d
            coverage = pred.feature_coverage
            regime = pred.regime

            # FIX: Quantile-based signal assignment
            if pred_5d >= q95 and prob_pos > 0.53:
                signal = "STRONG_BUY"
                conviction = min(0.7 + (prob_pos - 0.5) * 0.4, 0.95)
            elif pred_5d >= q85 and prob_pos > 0.51:
                signal = "BUY"
                conviction = 0.6 + (prob_pos - 0.5) * 0.3
            elif pred_5d <= q05 and prob_pos < 0.47:
                signal = "STRONG_SELL"
                conviction = min(0.7 + (0.5 - prob_pos) * 0.4, 0.95)
            elif pred_5d <= q15 and prob_pos < 0.49:
                signal = "SELL"
                conviction = 0.6 + (0.5 - prob_pos) * 0.3
            else:
                signal = "HOLD"
                conviction = 0.5

            # =====================================================================
            # REGIME-AWARE ADJUSTMENT (v2.2.0 enhancement)
            # The model is contrarian and works best in choppy markets
            # Downgrade confidence in trending markets (BULL/BEAR)
            # =====================================================================

            regime_note = ""
            if regime in UNFAVORABLE_REGIMES:
                # Model has negative IC in these regimes - reduce confidence
                conviction = max(0.35, conviction - 0.2)
                if signal in ["STRONG_BUY", "STRONG_SELL"]:
                    signal = "BUY" if "BUY" in signal else "SELL"
                regime_note = f" [⚠️ {regime} regime - model less reliable]"
            elif regime in FAVORABLE_REGIMES:
                # Model works well in these regimes - slight boost
                conviction = min(0.95, conviction + 0.05)
                regime_note = f" [✓ {regime} regime - model reliable]"

            # Confidence based on strength and coverage
            distance = abs(prob_pos - 0.5)
            if distance > 0.06 and abs(pred_5d) > 0.3 and coverage > 0.7:
                confidence = "HIGH"
            elif distance > 0.02 or abs(pred_5d) > 0.15:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            # FIX: Downgrade if low coverage
            if coverage < 0.6:
                confidence = "LOW"
                conviction = max(0.4, conviction - 0.15)

            # FIX: Downgrade confidence in unfavorable regimes
            if regime in UNFAVORABLE_REGIMES and confidence == "HIGH":
                confidence = "MEDIUM"

            results.append({
                'ticker': pred.ticker,
                'signal': signal,
                'conviction': conviction,
                'confidence': confidence,
                'expected_return_5d': pred.expected_return_5d,
                'expected_return_10d': pred.expected_return_10d,
                'ci_lower_5d': pred.ci_lower_5d,
                'ci_upper_5d': pred.ci_upper_5d,
                'prob_positive_5d': pred.prob_positive_5d,
                'prob_beat_market_5d': pred.prob_beat_market_5d,
                'sharpe_implied': pred.sharpe_ratio_implied,
                'position_size': pred.recommended_position_size,
                'regime': pred.regime,
                'regime_favorable': regime in FAVORABLE_REGIMES,
                'sector': pred.sector,
                'coverage': coverage,
                'top_factor': pred.top_bullish_factors[0][0] if pred.top_bullish_factors else 'N/A'
            })

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('expected_return_5d', ascending=False)

        # Log signal distribution with regime info
        signal_dist = result_df['signal'].value_counts()
        regime_dist = result_df['regime'].value_counts() if 'regime' in result_df.columns else {}
        favorable_count = result_df['regime_favorable'].sum() if 'regime_favorable' in result_df.columns else 0
        logger.info(f"Signal distribution: {signal_dist.to_dict()}")
        logger.info(f"Favorable regime signals: {favorable_count}/{len(result_df)}")

        return result_df

    def get_factor_report(self) -> pd.DataFrame:
        """Get detailed factor analysis report."""
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        records = []
        for factor in self.feature_names:
            weight_5d = self.global_weights.get(5, FactorWeights(
                MarketRegime.NEUTRAL, SectorGroup.OTHER, 5, {}, 0, 0, 0, 0, 0, date.today(), date.today()
            )).weights.get(factor, 0)

            weight_10d = self.global_weights.get(10, FactorWeights(
                MarketRegime.NEUTRAL, SectorGroup.OTHER, 10, {}, 0, 0, 0, 0, 0, date.today(), date.today()
            )).weights.get(factor, 0)

            defn = FACTOR_DEFINITIONS.get(factor)

            # FIX: Use permutation importance if available
            if factor in self._permutation_importance:
                importance = self._permutation_importance[factor]['mean']
                # Normalize
                total = sum(v['mean'] for v in self._permutation_importance.values()) or 1
                importance = importance / total
            else:
                importance = self.factor_importance.get(factor, 0)

            records.append({
                'factor': factor,
                'category': defn.category if defn else 'unknown',
                'weight_5d': weight_5d,
                'weight_10d': weight_10d,
                'importance': importance,
                'expected_sign': defn.expected_sign if defn else 0,
                'actual_sign': 1 if weight_5d > 0 else -1,
                'sign_match': (defn.expected_sign if defn else 0) * weight_5d > 0
            })

        df = pd.DataFrame(records)
        df = df.sort_values('importance', ascending=False)

        return df

    def save(self, path: str = 'models/multi_factor_alpha.pkl'):
        """Save model to file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        state = {
            'version': self.VERSION,
            'target_horizons': self.target_horizons,
            'min_samples': self.min_samples,
            'alpha': self.alpha,
            'use_regime_models': self.use_regime_models,
            'use_sector_models': self.use_sector_models,
            'feature_names': self.feature_names,
            'global_models': self.global_models,
            'global_weights': {k: asdict(v) for k, v in self.global_weights.items()},
            'models': self.models,
            'factor_weights': {str(k): asdict(v) for k, v in self.factor_weights.items()},
            'return_stats': self.return_stats,
            'factor_importance': self.factor_importance,
            'permutation_importance': self._permutation_importance,
            'validation_report': asdict(self.validation_report) if self.validation_report else None,
            'scaler': self.data_loader.scaler,
            'feature_medians': self.data_loader._feature_medians,
            'feature_stds': self.data_loader._feature_stds,
            'feature_coverage': self.data_loader._feature_coverage,
            # v2.1+ frozen schema
            'base_features': self.data_loader._base_features,
            'indicator_features': self.data_loader._indicator_features,
            'frozen_feature_schema': self.data_loader._frozen_feature_schema,
            'is_trained': self._is_trained
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Model saved to {path} (version {self.VERSION})")

    def load(self, path: str = 'models/multi_factor_alpha.pkl'):
        """Load model from file.

        Step 3 FIX: Properly enforce _is_trained based on loaded content,
        not just stored flag (handles old pickle files).
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.VERSION = state.get('version', '1.0.0')
        self.target_horizons = state.get('target_horizons', [5, 10, 20])
        self.min_samples = state.get('min_samples', 500)
        self.alpha = state.get('alpha', 0.1)
        self.use_regime_models = state.get('use_regime_models', True)
        self.use_sector_models = state.get('use_sector_models', True)
        self.feature_names = state.get('feature_names', [])
        self.global_models = state.get('global_models', {})
        self.models = state.get('models', {})
        self.return_stats = state.get('return_stats', {})
        self.factor_importance = state.get('factor_importance', {})
        self._permutation_importance = state.get('permutation_importance', {})

        # Restore scaler and feature stats
        if 'scaler' in state:
            self.data_loader.scaler = state['scaler']
        if 'feature_medians' in state:
            self.data_loader._feature_medians = state['feature_medians']
        if 'feature_stds' in state:
            self.data_loader._feature_stds = state['feature_stds']
        if 'feature_coverage' in state:
            self.data_loader._feature_coverage = state['feature_coverage']

        # Restore frozen schema if available (v2.1+ models)
        if 'base_features' in state:
            self.data_loader._base_features = state['base_features']
        if 'indicator_features' in state:
            self.data_loader._indicator_features = state['indicator_features']
        if 'frozen_feature_schema' in state:
            self.data_loader._frozen_feature_schema = state['frozen_feature_schema']

        # Restore global weights
        for k, v in state.get('global_weights', {}).items():
            horizon = int(k) if isinstance(k, str) else k
            # Handle both old 'ic' and new 'information_coefficient' keys
            ic_val = v.get('information_coefficient', v.get('ic', 0))
            self.global_weights[horizon] = FactorWeights(
                regime=MarketRegime(v.get('regime', 'NEUTRAL')) if isinstance(v.get('regime'), str) else v.get('regime', MarketRegime.NEUTRAL),
                sector_group=SectorGroup(v.get('sector_group', 'OTHER')) if isinstance(v.get('sector_group'), str) else v.get('sector_group', SectorGroup.OTHER),
                horizon_days=v.get('horizon_days', horizon),
                weights=v.get('weights', {}),
                intercept=v.get('intercept', 0),
                r_squared=v.get('r_squared', 0),
                mse=v.get('mse', 0),
                information_coefficient=ic_val,
                n_samples=v.get('n_samples', 0),
                train_start=v.get('train_start', date.today()),
                train_end=v.get('train_end', date.today())
            )

        # Restore validation report
        vr = state.get('validation_report')
        if vr:
            self.validation_report = ModelValidationReport(
                model_version=vr.get('model_version', '1.0.0'),
                trained_at=vr.get('trained_at', datetime.now()),
                overall_ic=float(vr.get('overall_ic', 0)),
                overall_icir=float(vr.get('overall_icir', 0)),
                overall_r2=float(vr.get('overall_r2', 0)),
                regime_performance=vr.get('regime_performance', {}),
                sector_performance=vr.get('sector_performance', {}),
                n_folds=vr.get('n_folds', 0),
                fold_results=vr.get('fold_results', []),
                mean_ic_oos=float(vr.get('mean_ic_oos', 0)),
                std_ic_oos=float(vr.get('std_ic_oos', 0)),
                factor_importance=vr.get('factor_importance', {}),
                factor_stability=vr.get('factor_stability', {}),
                max_drawdown_from_signals=float(vr.get('max_drawdown_from_signals', 0)),
                turnover=float(vr.get('turnover', 0)),
                ic_tstat=float(vr.get('ic_tstat', 0)),
                ic_pvalue=float(vr.get('ic_pvalue', 1)),
                beats_baseline=bool(vr.get('beats_baseline', False))
            )

        # FIX Step 3: Enforce _is_trained based on actual loaded content
        # Don't rely on stored flag - check if we have what we need
        self._is_trained = bool(self.global_models) and bool(self.feature_names)

        if not self._is_trained:
            logger.warning(f"Model loaded but appears untrained (global_models={len(self.global_models)}, features={len(self.feature_names)})")

        logger.info(f"Model loaded from {path} (version {self.VERSION}, trained={self._is_trained})")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_alpha_model(path: str = 'models/multi_factor_alpha.pkl') -> MultiFactorAlphaModel:
    """Load a trained alpha model."""
    model = MultiFactorAlphaModel()
    model.load(path)
    return model


def train_alpha_model(min_date: str = None,
                      save_path: str = 'models/multi_factor_alpha.pkl') -> ModelValidationReport:
    """Train and save a new alpha model."""
    model = MultiFactorAlphaModel()
    report = model.train(min_date=min_date)
    model.save(save_path)
    return report


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-FACTOR ALPHA MODEL v2.0 (WITH FIXES)")
    print("=" * 70)

    # Train model
    report = train_alpha_model(min_date='2024-06-01')

    # Test predictions
    model = load_alpha_model()
    predictions = model.predict_live()

    # Focus on favorable regime signals
    good_signals = predictions[
        (predictions['regime_favorable'] == True) &
        (predictions['signal'] != 'HOLD')
        ]
    print(f"\nGenerated {len(predictions)} predictions")
    print(f"\nSignal distribution:")
    print(predictions['signal'].value_counts())
    print(f"\nConfidence distribution:")
    print(predictions['confidence'].value_counts())

    print("\nTop 10 predictions:")
    print(predictions.head(10)[['ticker', 'signal', 'confidence', 'expected_return_5d', 'prob_positive_5d']])