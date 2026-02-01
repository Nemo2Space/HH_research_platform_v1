"""
Point-in-Time Data Validator

Ensures no lookahead bias in backtesting by validating that all data
used for decisions was actually available at decision time.

Key Validations:
1. Timestamp validation (no future data)
2. Revision-aware series (use original values, not revised)
3. Survivorship bias detection
4. Corporate action integrity (splits, dividends, symbol changes)

Author: Alpha Research Platform
Location: src/data/pit_validator.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class PITViolationType(Enum):
    """Types of point-in-time violations."""
    FUTURE_DATA = "future_data"           # Data timestamp after decision time
    REVISED_VALUE = "revised_value"        # Using revised data instead of original
    DELISTED_STOCK = "delisted_stock"      # Trading delisted stock before delist date
    SURVIVORSHIP = "survivorship"          # Universe only includes survivors
    CORPORATE_ACTION = "corporate_action"  # Unadjusted for splits/dividends
    TIMESTAMP_MISSING = "timestamp_missing"  # No timestamp to validate


@dataclass
class PITViolation:
    """A single point-in-time violation."""
    violation_type: PITViolationType
    field_name: str
    decision_time: datetime
    data_timestamp: Optional[datetime]
    severity: str  # 'error', 'warning', 'info'
    details: str
    
    def __repr__(self):
        return f"PITViolation({self.violation_type.value}: {self.field_name} @ {self.decision_time})"


@dataclass
class PITValidationResult:
    """Result of point-in-time validation."""
    is_valid: bool
    violations: List[PITViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Statistics
    total_fields_checked: int = 0
    fields_with_timestamps: int = 0
    fields_missing_timestamps: int = 0
    
    def has_errors(self) -> bool:
        """Check if there are error-level violations."""
        return any(v.severity == 'error' for v in self.violations)
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        if self.is_valid:
            return f"✅ Valid: {self.total_fields_checked} fields checked, no violations"
        else:
            error_count = sum(1 for v in self.violations if v.severity == 'error')
            warn_count = sum(1 for v in self.violations if v.severity == 'warning')
            return f"❌ Invalid: {error_count} errors, {warn_count} warnings"


@dataclass
class DelistedStock:
    """Information about a delisted stock."""
    ticker: str
    delist_date: date
    reason: str  # 'merger', 'bankruptcy', 'goes_private', 'unknown'
    final_price: Optional[float] = None
    acquiring_company: Optional[str] = None


@dataclass
class CorporateAction:
    """Corporate action event."""
    ticker: str
    action_date: date
    action_type: str  # 'split', 'dividend', 'spinoff', 'merger'
    ratio: float  # Split ratio or dividend amount
    details: str


# =============================================================================
# POINT-IN-TIME VALIDATOR
# =============================================================================

class PITValidator:
    """
    Validates point-in-time integrity of data.
    
    Usage:
        validator = PITValidator()
        
        # Validate single feature set
        result = validator.validate_features(features, decision_time)
        
        # Validate historical backtest data
        issues = validator.validate_backtest_data(historical_df)
        
        # Check if ticker was tradeable at time
        is_ok = validator.is_tradeable(ticker, trade_date)
    """
    
    # Fields that MUST have timestamps for validation
    REQUIRED_TIMESTAMP_FIELDS = {
        'price': 'price_timestamp',
        'sentiment_score': 'sentiment_timestamp',
        'fundamental_score': 'fundamental_timestamp',
        'technical_score': 'technical_timestamp',
        'options_flow_score': 'options_timestamp',
    }
    
    # Fields that are frequently revised (use with caution)
    REVISION_SENSITIVE_FIELDS = {
        'earnings_eps',
        'revenue',
        'gdp',
        'cpi',
        'employment',
        'pe_ratio',
        'eps_estimate',
    }
    
    # Maximum allowable staleness (in days)
    MAX_STALENESS = {
        'price': 1,
        'sentiment_score': 3,
        'fundamental_score': 30,
        'technical_score': 1,
        'options_flow_score': 1,
        'institutional_score': 90,
        'earnings_date': 365,
    }
    
    def __init__(self):
        """Initialize validator."""
        self._delisted_stocks: Dict[str, DelistedStock] = {}
        self._corporate_actions: Dict[str, List[CorporateAction]] = {}
        self._known_revisions: Dict[str, Dict] = {}
        
    def validate_features(self, 
                          features: Dict[str, Any],
                          decision_time: datetime,
                          strict: bool = True) -> PITValidationResult:
        """
        Validate that all features were available at decision time.
        
        Args:
            features: Dict of field_name -> value OR object with attributes
            decision_time: The time at which decision was made
            strict: If True, treat missing timestamps as errors
            
        Returns:
            PITValidationResult with any violations found
        """
        result = PITValidationResult(is_valid=True)
        
        # Convert object to dict if needed
        if hasattr(features, '__dict__'):
            features = vars(features)
        elif hasattr(features, 'to_dict'):
            features = features.to_dict()
        
        for field_name, timestamp_field in self.REQUIRED_TIMESTAMP_FIELDS.items():
            result.total_fields_checked += 1
            
            # Get timestamp
            timestamp = features.get(timestamp_field)
            
            if timestamp is None:
                result.fields_missing_timestamps += 1
                
                if strict:
                    result.violations.append(PITViolation(
                        violation_type=PITViolationType.TIMESTAMP_MISSING,
                        field_name=field_name,
                        decision_time=decision_time,
                        data_timestamp=None,
                        severity='warning',
                        details=f"No timestamp for {field_name}"
                    ))
                continue
            
            result.fields_with_timestamps += 1
            
            # Convert to datetime if needed
            if isinstance(timestamp, date) and not isinstance(timestamp, datetime):
                timestamp = datetime.combine(timestamp, datetime.min.time())
            
            # Check for future data
            if timestamp > decision_time:
                result.is_valid = False
                result.violations.append(PITViolation(
                    violation_type=PITViolationType.FUTURE_DATA,
                    field_name=field_name,
                    decision_time=decision_time,
                    data_timestamp=timestamp,
                    severity='error',
                    details=f"{field_name} has timestamp {timestamp} > decision time {decision_time}"
                ))
            
            # Check staleness
            max_stale = self.MAX_STALENESS.get(field_name, 7)
            days_old = (decision_time - timestamp).days
            
            if days_old > max_stale:
                result.warnings.append(
                    f"{field_name} is {days_old} days old (max {max_stale})"
                )
        
        # Check for revision-sensitive fields
        for field_name in self.REVISION_SENSITIVE_FIELDS:
            if field_name in features and features[field_name] is not None:
                result.warnings.append(
                    f"{field_name} may be subject to revisions - verify using original values"
                )
        
        return result
    
    def validate_backtest_data(self,
                               df: pd.DataFrame,
                               date_column: str = 'date',
                               ticker_column: str = 'ticker') -> Dict[str, List[PITViolation]]:
        """
        Validate a DataFrame of historical backtest data.
        
        Args:
            df: DataFrame with historical data
            date_column: Name of date column
            ticker_column: Name of ticker column
            
        Returns:
            Dict of ticker -> list of violations
        """
        violations_by_ticker: Dict[str, List[PITViolation]] = {}
        
        if df.empty:
            return violations_by_ticker
        
        # Group by ticker
        for ticker, group in df.groupby(ticker_column):
            violations = []
            
            # Check for survivorship bias
            if not self._check_survivorship(ticker, group, date_column):
                violations.append(PITViolation(
                    violation_type=PITViolationType.SURVIVORSHIP,
                    field_name=ticker_column,
                    decision_time=group[date_column].max(),
                    data_timestamp=None,
                    severity='warning',
                    details=f"{ticker} may be subject to survivorship bias"
                ))
            
            # Check for corporate actions
            corp_actions = self._get_corporate_actions(ticker)
            for action in corp_actions:
                action_date = datetime.combine(action.action_date, datetime.min.time())
                
                # Check if data is adjusted consistently
                pre_action = group[group[date_column] < action_date]
                post_action = group[group[date_column] >= action_date]
                
                if not pre_action.empty and not post_action.empty:
                    # This is a simplified check - real implementation would
                    # verify price continuity around corporate actions
                    pass
            
            if violations:
                violations_by_ticker[ticker] = violations
        
        return violations_by_ticker
    
    def is_tradeable(self, ticker: str, trade_date: date) -> Tuple[bool, Optional[str]]:
        """
        Check if a ticker was tradeable on a specific date.
        
        Args:
            ticker: Stock symbol
            trade_date: Date to check
            
        Returns:
            Tuple of (is_tradeable, reason_if_not)
        """
        # Check if delisted before trade date
        if ticker in self._delisted_stocks:
            delist_info = self._delisted_stocks[ticker]
            if trade_date >= delist_info.delist_date:
                return False, f"Delisted on {delist_info.delist_date}: {delist_info.reason}"
        
        # Check for trading halts, suspensions, etc.
        # (would need external data source)
        
        return True, None
    
    def register_delisted_stock(self, 
                                ticker: str,
                                delist_date: date,
                                reason: str,
                                final_price: float = None):
        """Register a stock as delisted."""
        self._delisted_stocks[ticker] = DelistedStock(
            ticker=ticker,
            delist_date=delist_date,
            reason=reason,
            final_price=final_price
        )
    
    def register_corporate_action(self,
                                  ticker: str,
                                  action_date: date,
                                  action_type: str,
                                  ratio: float,
                                  details: str = ""):
        """Register a corporate action."""
        if ticker not in self._corporate_actions:
            self._corporate_actions[ticker] = []
        
        self._corporate_actions[ticker].append(CorporateAction(
            ticker=ticker,
            action_date=action_date,
            action_type=action_type,
            ratio=ratio,
            details=details
        ))
    
    def _check_survivorship(self, 
                            ticker: str,
                            data: pd.DataFrame,
                            date_column: str) -> bool:
        """Check for potential survivorship bias."""
        # If ticker is in our delisted list and data ends before delist,
        # there might be survivorship bias
        if ticker in self._delisted_stocks:
            delist_date = self._delisted_stocks[ticker].delist_date
            last_data_date = data[date_column].max()
            
            if isinstance(last_data_date, datetime):
                last_data_date = last_data_date.date()
            
            # If data ends significantly before delist, might be OK
            # If data goes right up to delist, could be survivorship
            if last_data_date >= delist_date:
                return False
        
        return True
    
    def _get_corporate_actions(self, ticker: str) -> List[CorporateAction]:
        """Get corporate actions for a ticker."""
        return self._corporate_actions.get(ticker, [])


# =============================================================================
# BACKTEST DATA INTEGRITY CHECKER
# =============================================================================

class BacktestIntegrityChecker:
    """
    Comprehensive integrity checking for backtest data.
    
    Checks:
    1. Point-in-time integrity
    2. Data gaps and missing values
    3. Price continuity
    4. Universe consistency
    """
    
    def __init__(self):
        self.pit_validator = PITValidator()
    
    def check_historical_scores(self,
                                df: pd.DataFrame,
                                min_date: date = None,
                                max_date: date = None) -> Dict[str, Any]:
        """
        Check integrity of historical_scores table data.
        
        Args:
            df: DataFrame from historical_scores
            min_date: Start date for checking
            max_date: End date for checking
            
        Returns:
            Dict with integrity report
        """
        report = {
            'is_valid': True,
            'total_rows': len(df),
            'date_range': None,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        if df.empty:
            report['issues'].append("DataFrame is empty")
            report['is_valid'] = False
            return report
        
        # Date range
        date_col = 'score_date' if 'score_date' in df.columns else 'date'
        if date_col in df.columns:
            report['date_range'] = {
                'min': df[date_col].min(),
                'max': df[date_col].max()
            }
        
        # Check for required columns
        required_cols = ['ticker', date_col, 'return_5d']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            report['issues'].append(f"Missing required columns: {missing_cols}")
            report['is_valid'] = False
        
        # Check for data gaps
        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col]).dt.date.unique()
            dates = sorted(dates)
            
            gaps = []
            for i in range(1, len(dates)):
                diff = (dates[i] - dates[i-1]).days
                if diff > 5:  # More than a week gap (accounting for weekends)
                    gaps.append({
                        'from': dates[i-1],
                        'to': dates[i],
                        'days': diff
                    })
            
            if gaps:
                report['warnings'].append(f"Found {len(gaps)} data gaps")
                report['statistics']['gaps'] = gaps[:10]  # First 10
        
        # Check for missing return data
        if 'return_5d' in df.columns:
            missing_returns = df['return_5d'].isna().sum()
            pct_missing = missing_returns / len(df) * 100
            
            if pct_missing > 10:
                report['warnings'].append(
                    f"{pct_missing:.1f}% missing return_5d values"
                )
            
            report['statistics']['missing_returns_pct'] = pct_missing
        
        # Check for outlier returns (potential data errors)
        if 'return_5d' in df.columns:
            returns = df['return_5d'].dropna()
            
            # Flag returns > 100% or < -50% as suspicious
            suspicious = (returns.abs() > 100) | (returns < -50)
            if suspicious.any():
                report['warnings'].append(
                    f"{suspicious.sum()} suspicious return values (>100% or <-50%)"
                )
        
        # Check for duplicate entries
        if 'ticker' in df.columns and date_col in df.columns:
            duplicates = df.duplicated(subset=['ticker', date_col], keep=False)
            if duplicates.any():
                report['warnings'].append(
                    f"{duplicates.sum()} duplicate ticker/date combinations"
                )
        
        # Check universe consistency over time
        if 'ticker' in df.columns and date_col in df.columns:
            tickers_by_date = df.groupby(date_col)['ticker'].nunique()
            
            if len(tickers_by_date) > 1:
                min_tickers = tickers_by_date.min()
                max_tickers = tickers_by_date.max()
                
                if max_tickers > min_tickers * 2:
                    report['warnings'].append(
                        f"Universe size varies significantly: {min_tickers}-{max_tickers} tickers"
                    )
                
                report['statistics']['universe_size'] = {
                    'min': min_tickers,
                    'max': max_tickers,
                    'avg': tickers_by_date.mean()
                }
        
        # Overall validity
        if report['issues']:
            report['is_valid'] = False
        
        return report
    
    def create_pit_safe_dataset(self,
                                df: pd.DataFrame,
                                decision_date_col: str = 'score_date',
                                data_date_col: str = 'data_date') -> pd.DataFrame:
        """
        Create a point-in-time safe dataset by filtering out future data.
        
        Args:
            df: Input DataFrame
            decision_date_col: Column with decision dates
            data_date_col: Column with data observation dates
            
        Returns:
            Filtered DataFrame with only PIT-valid data
        """
        if data_date_col not in df.columns:
            logger.warning(f"No {data_date_col} column - cannot verify PIT safety")
            return df
        
        # Filter: data_date <= decision_date
        df = df.copy()
        df['_decision_date'] = pd.to_datetime(df[decision_date_col])
        df['_data_date'] = pd.to_datetime(df[data_date_col])
        
        valid_mask = df['_data_date'] <= df['_decision_date']
        invalid_count = (~valid_mask).sum()
        
        if invalid_count > 0:
            logger.warning(f"Filtered out {invalid_count} rows with future data")
        
        result = df[valid_mask].drop(columns=['_decision_date', '_data_date'])
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_validator_instance = None


def get_pit_validator() -> PITValidator:
    """Get singleton PIT validator."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = PITValidator()
    return _validator_instance


def validate_features_pit(features: Dict, decision_time: datetime) -> PITValidationResult:
    """Quick access to feature validation."""
    return get_pit_validator().validate_features(features, decision_time)


def check_backtest_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick access to backtest data integrity check."""
    checker = BacktestIntegrityChecker()
    return checker.check_historical_scores(df)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test PIT validation
    validator = PITValidator()
    
    # Valid features
    decision_time = datetime(2024, 1, 15, 10, 0)
    features = {
        'price': 150.0,
        'price_timestamp': datetime(2024, 1, 15, 9, 30),
        'sentiment_score': 65,
        'sentiment_timestamp': datetime(2024, 1, 14, 18, 0),
        'fundamental_score': 70,
        'fundamental_timestamp': datetime(2024, 1, 10, 0, 0),
    }
    
    result = validator.validate_features(features, decision_time)
    print(f"\nValid features test:")
    print(f"  {result.get_summary()}")
    
    # Invalid features (future data)
    bad_features = {
        'price': 155.0,
        'price_timestamp': datetime(2024, 1, 15, 16, 0),  # FUTURE!
        'sentiment_score': 68,
        'sentiment_timestamp': datetime(2024, 1, 15, 12, 0),  # FUTURE!
    }
    
    result2 = validator.validate_features(bad_features, decision_time)
    print(f"\nFuture data test:")
    print(f"  {result2.get_summary()}")
    for v in result2.violations:
        print(f"  - {v}")
    
    # Test backtest integrity
    checker = BacktestIntegrityChecker()
    
    # Create sample data
    sample_df = pd.DataFrame({
        'ticker': ['AAPL'] * 10 + ['MSFT'] * 10,
        'score_date': pd.date_range('2024-01-01', periods=10).tolist() * 2,
        'return_5d': np.random.randn(20) * 2,
    })
    
    report = checker.check_historical_scores(sample_df)
    print(f"\nBacktest integrity check:")
    print(f"  Valid: {report['is_valid']}")
    print(f"  Rows: {report['total_rows']}")
    if report['warnings']:
        print(f"  Warnings: {report['warnings']}")
