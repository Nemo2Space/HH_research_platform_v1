"""
Feature Logger - Deterministic Replay Support

Stores all inputs (features) and outputs (scores) for every scoring run
to enable:
1. Exact reproduction of any historical signal
2. Debugging signal discrepancies
3. Model version tracking
4. Audit trail for compliance

Author: Alpha Research Platform
Location: src/core/feature_logger.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import json
import hashlib
import gzip
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScoringRun:
    """Complete record of a single scoring run."""
    # Identifiers
    run_id: str
    ticker: str
    as_of_time: datetime
    
    # Version info
    scorer_version: str
    model_version: str = "1.0.0"
    
    # All input features (serialized)
    features_json: str = ""
    feature_hash: str = ""
    
    # All output scores
    scores_json: str = ""
    
    # Data quality
    data_quality_json: str = ""
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    environment: str = "production"  # production, backtest, test
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['as_of_time'] = self.as_of_time.isoformat()
        result['created_at'] = self.created_at.isoformat()
        return result


class FeatureLogger:
    """
    Logs all scoring inputs/outputs for reproducibility.
    
    Usage:
        logger = FeatureLogger(storage_path='./logs/features')
        
        # Log a scoring run
        logger.log_run(ticker, as_of_time, features, scores)
        
        # Retrieve for replay
        run = logger.get_run(run_id)
        features = json.loads(run.features_json)
    """
    
    def __init__(self, 
                 storage_path: str = './logs/features',
                 db_engine=None,
                 compress: bool = True):
        """
        Initialize feature logger.
        
        Args:
            storage_path: Path for file-based logging
            db_engine: SQLAlchemy engine for DB logging (optional)
            compress: Whether to gzip log files
        """
        self.storage_path = storage_path
        self.db_engine = db_engine
        self.compress = compress
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # In-memory buffer for batch writes
        self._buffer: List[ScoringRun] = []
        self._buffer_size = 100
    
    def log_run(self,
                ticker: str,
                as_of_time: datetime,
                features: Any,
                scores: Any,
                scorer_version: str = "1.0.0",
                environment: str = "production") -> str:
        """
        Log a complete scoring run.
        
        Args:
            ticker: Stock symbol
            as_of_time: Decision time
            features: TickerFeatures object or dict
            scores: ScoringResult object or dict
            scorer_version: Version of scorer used
            environment: production, backtest, or test
            
        Returns:
            run_id for retrieval
        """
        # Convert objects to dicts
        if hasattr(features, 'to_dict'):
            features_dict = features.to_dict()
        elif hasattr(features, '__dict__'):
            features_dict = self._serialize_object(features)
        else:
            features_dict = features
        
        if hasattr(scores, 'to_dict'):
            scores_dict = scores.to_dict()
        elif hasattr(scores, '__dict__'):
            scores_dict = self._serialize_object(scores)
        else:
            scores_dict = scores
        
        # Extract data quality
        data_quality = {}
        if hasattr(features, 'data_quality') and features.data_quality:
            if hasattr(features.data_quality, 'to_dict'):
                data_quality = features.data_quality.to_dict()
        
        # Generate feature hash
        feature_hash = self._compute_hash(features_dict)
        
        # Generate run ID
        run_id = f"{ticker}_{as_of_time.strftime('%Y%m%d_%H%M%S')}_{feature_hash[:8]}"
        
        # Create run record
        run = ScoringRun(
            run_id=run_id,
            ticker=ticker,
            as_of_time=as_of_time,
            scorer_version=scorer_version,
            features_json=json.dumps(features_dict, default=str),
            feature_hash=feature_hash,
            scores_json=json.dumps(scores_dict, default=str),
            data_quality_json=json.dumps(data_quality, default=str),
            environment=environment,
        )
        
        # Add to buffer
        self._buffer.append(run)
        
        # Flush if buffer full
        if len(self._buffer) >= self._buffer_size:
            self.flush()
        
        return run_id
    
    def flush(self):
        """Write buffered runs to storage."""
        if not self._buffer:
            return
        
        # Write to database if available
        if self.db_engine:
            self._write_to_db(self._buffer)
        
        # Write to files
        self._write_to_files(self._buffer)
        
        self._buffer = []
    
    def get_run(self, run_id: str) -> Optional[ScoringRun]:
        """Retrieve a specific scoring run."""
        # Try database first
        if self.db_engine:
            run = self._load_from_db(run_id)
            if run:
                return run
        
        # Try files
        return self._load_from_file(run_id)
    
    def get_runs_for_ticker(self, 
                            ticker: str,
                            start_date: datetime = None,
                            end_date: datetime = None,
                            limit: int = 100) -> List[ScoringRun]:
        """Get all scoring runs for a ticker in a date range."""
        runs = []
        
        if self.db_engine:
            runs = self._query_db(ticker, start_date, end_date, limit)
        
        return runs
    
    def replay_scoring(self, run_id: str) -> Dict:
        """
        Replay a scoring run to verify reproducibility.
        
        Returns dict with original and replayed scores for comparison.
        """
        run = self.get_run(run_id)
        if not run:
            return {'error': f'Run {run_id} not found'}
        
        # Parse original features
        original_features = json.loads(run.features_json)
        original_scores = json.loads(run.scores_json)
        
        # Would need to import UnifiedScorer and replay
        # For now, just return the original data
        return {
            'run_id': run_id,
            'ticker': run.ticker,
            'as_of_time': run.as_of_time.isoformat(),
            'original_features': original_features,
            'original_scores': original_scores,
            'scorer_version': run.scorer_version,
            'feature_hash': run.feature_hash,
        }
    
    def compare_runs(self, run_id_1: str, run_id_2: str) -> Dict:
        """Compare two scoring runs to identify differences."""
        run1 = self.get_run(run_id_1)
        run2 = self.get_run(run_id_2)
        
        if not run1 or not run2:
            return {'error': 'One or both runs not found'}
        
        features1 = json.loads(run1.features_json)
        features2 = json.loads(run2.features_json)
        scores1 = json.loads(run1.scores_json)
        scores2 = json.loads(run2.scores_json)
        
        # Find feature differences
        feature_diffs = []
        all_keys = set(features1.keys()) | set(features2.keys())
        
        for key in all_keys:
            v1 = features1.get(key)
            v2 = features2.get(key)
            if v1 != v2:
                feature_diffs.append({
                    'field': key,
                    'run1': v1,
                    'run2': v2
                })
        
        # Find score differences
        score_diffs = []
        all_score_keys = set(scores1.keys()) | set(scores2.keys())
        
        for key in all_score_keys:
            v1 = scores1.get(key)
            v2 = scores2.get(key)
            if v1 != v2:
                score_diffs.append({
                    'field': key,
                    'run1': v1,
                    'run2': v2
                })
        
        return {
            'run1': run_id_1,
            'run2': run_id_2,
            'feature_differences': feature_diffs,
            'score_differences': score_diffs,
            'same_features': len(feature_diffs) == 0,
            'same_scores': len(score_diffs) == 0,
        }
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _serialize_object(self, obj: Any) -> Dict:
        """Serialize object to dict, handling special types."""
        result = {}
        
        for key, value in obj.__dict__.items():
            if key.startswith('_'):
                continue
            
            if isinstance(value, (datetime, date)):
                result[key] = value.isoformat()
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                result[key] = float(value)
            elif hasattr(value, '__dict__'):
                result[key] = self._serialize_object(value)
            elif isinstance(value, list):
                result[key] = [
                    self._serialize_object(v) if hasattr(v, '__dict__') else v
                    for v in value
                ]
            else:
                result[key] = value
        
        return result
    
    def _compute_hash(self, features: Dict) -> str:
        """Compute deterministic hash of features."""
        # Sort keys for determinism
        sorted_str = json.dumps(features, sort_keys=True, default=str)
        return hashlib.sha256(sorted_str.encode()).hexdigest()[:16]
    
    def _write_to_db(self, runs: List[ScoringRun]):
        """Write runs to database."""
        try:
            df = pd.DataFrame([r.to_dict() for r in runs])
            df.to_sql(
                'scoring_runs',
                self.db_engine,
                if_exists='append',
                index=False
            )
        except Exception as e:
            logger.error(f"Error writing to database: {e}")
    
    def _write_to_files(self, runs: List[ScoringRun]):
        """Write runs to daily log files."""
        # Group by date
        by_date = {}
        for run in runs:
            date_str = run.as_of_time.strftime('%Y%m%d')
            if date_str not in by_date:
                by_date[date_str] = []
            by_date[date_str].append(run.to_dict())
        
        # Write each date's runs
        for date_str, date_runs in by_date.items():
            filename = f"scoring_runs_{date_str}.jsonl"
            if self.compress:
                filename += '.gz'
            
            filepath = os.path.join(self.storage_path, filename)
            
            try:
                if self.compress:
                    with gzip.open(filepath, 'at') as f:
                        for run in date_runs:
                            f.write(json.dumps(run, default=str) + '\n')
                else:
                    with open(filepath, 'a') as f:
                        for run in date_runs:
                            f.write(json.dumps(run, default=str) + '\n')
            except Exception as e:
                logger.error(f"Error writing to file: {e}")
    
    def _load_from_db(self, run_id: str) -> Optional[ScoringRun]:
        """Load run from database."""
        try:
            query = "SELECT * FROM scoring_runs WHERE run_id = %(run_id)s"
            df = pd.read_sql(query, self.db_engine, params={'run_id': run_id})
            
            if len(df) > 0:
                row = df.iloc[0].to_dict()
                return ScoringRun(
                    run_id=row['run_id'],
                    ticker=row['ticker'],
                    as_of_time=pd.to_datetime(row['as_of_time']),
                    scorer_version=row['scorer_version'],
                    features_json=row['features_json'],
                    feature_hash=row['feature_hash'],
                    scores_json=row['scores_json'],
                    data_quality_json=row.get('data_quality_json', '{}'),
                    environment=row.get('environment', 'production'),
                )
        except Exception as e:
            logger.debug(f"Error loading from database: {e}")
        return None
    
    def _load_from_file(self, run_id: str) -> Optional[ScoringRun]:
        """Load run from file by searching log files."""
        # Extract date from run_id (format: TICKER_YYYYMMDD_HHMMSS_HASH)
        parts = run_id.split('_')
        if len(parts) >= 2:
            date_str = parts[1]
            
            filename = f"scoring_runs_{date_str}.jsonl"
            if self.compress:
                filename += '.gz'
            
            filepath = os.path.join(self.storage_path, filename)
            
            if os.path.exists(filepath):
                try:
                    opener = gzip.open if self.compress else open
                    with opener(filepath, 'rt') as f:
                        for line in f:
                            run_dict = json.loads(line)
                            if run_dict.get('run_id') == run_id:
                                return ScoringRun(
                                    run_id=run_dict['run_id'],
                                    ticker=run_dict['ticker'],
                                    as_of_time=datetime.fromisoformat(run_dict['as_of_time']),
                                    scorer_version=run_dict['scorer_version'],
                                    features_json=run_dict['features_json'],
                                    feature_hash=run_dict['feature_hash'],
                                    scores_json=run_dict['scores_json'],
                                    data_quality_json=run_dict.get('data_quality_json', '{}'),
                                    environment=run_dict.get('environment', 'production'),
                                )
                except Exception as e:
                    logger.debug(f"Error reading file: {e}")
        
        return None
    
    def _query_db(self, 
                  ticker: str,
                  start_date: datetime = None,
                  end_date: datetime = None,
                  limit: int = 100) -> List[ScoringRun]:
        """Query runs from database."""
        runs = []
        
        try:
            query = "SELECT * FROM scoring_runs WHERE ticker = %(ticker)s"
            params = {'ticker': ticker}
            
            if start_date:
                query += " AND as_of_time >= %(start_date)s"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND as_of_time <= %(end_date)s"
                params['end_date'] = end_date
            
            query += f" ORDER BY as_of_time DESC LIMIT {limit}"
            
            df = pd.read_sql(query, self.db_engine, params=params)
            
            for _, row in df.iterrows():
                runs.append(ScoringRun(
                    run_id=row['run_id'],
                    ticker=row['ticker'],
                    as_of_time=pd.to_datetime(row['as_of_time']),
                    scorer_version=row['scorer_version'],
                    features_json=row['features_json'],
                    feature_hash=row['feature_hash'],
                    scores_json=row['scores_json'],
                    data_quality_json=row.get('data_quality_json', '{}'),
                    environment=row.get('environment', 'production'),
                ))
        except Exception as e:
            logger.debug(f"Error querying database: {e}")
        
        return runs


# =============================================================================
# DATABASE SCHEMA
# =============================================================================

SCORING_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS scoring_runs (
    run_id VARCHAR(100) PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    as_of_time TIMESTAMP NOT NULL,
    scorer_version VARCHAR(20),
    model_version VARCHAR(20),
    features_json TEXT,
    feature_hash VARCHAR(64),
    scores_json TEXT,
    data_quality_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    environment VARCHAR(20) DEFAULT 'production',
    
    -- Indexes
    INDEX idx_ticker_date (ticker, as_of_time),
    INDEX idx_feature_hash (feature_hash)
);
"""


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_logger_instance = None


def get_feature_logger(storage_path: str = './logs/features',
                       db_engine=None) -> FeatureLogger:
    """Get singleton feature logger."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = FeatureLogger(storage_path, db_engine)
    return _logger_instance


def log_scoring_run(ticker: str,
                    as_of_time: datetime,
                    features: Any,
                    scores: Any,
                    **kwargs) -> str:
    """Quick access to log a scoring run."""
    return get_feature_logger().log_run(ticker, as_of_time, features, scores, **kwargs)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test the feature logger
    logger_inst = FeatureLogger(storage_path='/tmp/feature_logs')
    
    # Mock features and scores
    features = {
        'ticker': 'AAPL',
        'current_price': 185.50,
        'sentiment_score': 65,
        'fundamental_score': 72,
        'technical_score': 58,
    }
    
    scores = {
        'total_score': 68,
        'signal_type': 'BUY',
        'confidence': 0.85,
    }
    
    # Log a run
    run_id = logger_inst.log_run(
        ticker='AAPL',
        as_of_time=datetime.now(),
        features=features,
        scores=scores,
        environment='test'
    )
    
    print(f"Logged run: {run_id}")
    
    # Flush to storage
    logger_inst.flush()
    
    # Retrieve
    run = logger_inst.get_run(run_id)
    if run:
        print(f"Retrieved: {run.ticker} @ {run.as_of_time}")
        print(f"Feature hash: {run.feature_hash}")
