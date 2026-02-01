"""
Signal Drift Monitor

Monitors for distribution shifts and anomalies in signals that
could indicate:
1. Data provider outages
2. Model degradation
3. Market regime changes
4. Data quality issues

Author: Alpha Research Platform
Location: src/monitoring/drift_monitor.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DriftType(Enum):
    """Types of drift detected."""
    DISTRIBUTION_SHIFT = "distribution_shift"
    MISSING_DATA = "missing_data"
    STALE_DATA = "stale_data"
    OUTLIER_SPIKE = "outlier_spike"
    CORRELATION_BREAK = "correlation_break"
    COVERAGE_DROP = "coverage_drop"


@dataclass
class DriftAlert:
    """A single drift alert."""
    alert_id: str
    drift_type: DriftType
    severity: AlertSeverity
    metric_name: str
    current_value: float
    expected_range: Tuple[float, float]
    message: str
    detected_at: datetime = field(default_factory=datetime.now)
    
    def __repr__(self):
        return f"DriftAlert({self.severity.value}: {self.metric_name} - {self.message})"


@dataclass
class MetricStats:
    """Running statistics for a metric."""
    name: str
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    last_value: float = 0.0
    last_updated: Optional[datetime] = None
    
    # Rolling window for recent values
    recent_values: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, value: float, timestamp: datetime = None):
        """Update running statistics with new value."""
        if value is None or np.isnan(value):
            return
        
        self.count += 1
        self.last_value = value
        self.last_updated = timestamp or datetime.now()
        self.recent_values.append(value)
        
        # Update min/max
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        
        # Welford's online algorithm for mean and variance
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        
        if self.count > 1:
            # Update variance
            m2 = (self.std ** 2) * (self.count - 1)
            m2 += delta * delta2
            self.std = np.sqrt(m2 / self.count)
    
    def get_z_score(self, value: float) -> float:
        """Get z-score for a value."""
        if self.std == 0 or self.count < 10:
            return 0.0
        return (value - self.mean) / self.std
    
    def get_percentile(self, value: float) -> float:
        """Get approximate percentile for a value."""
        if not self.recent_values:
            return 50.0
        values = sorted(self.recent_values)
        count_below = sum(1 for v in values if v < value)
        return (count_below / len(values)) * 100


class DriftMonitor:
    """
    Monitors signal distributions for drift and anomalies.
    
    Usage:
        monitor = DriftMonitor()
        
        # Record metrics during scoring
        monitor.record_metric('sentiment_score', 65, ticker='AAPL')
        monitor.record_metric('article_count', 12, ticker='AAPL')
        
        # Check for drift
        alerts = monitor.check_drift()
        
        # Get monitoring report
        report = monitor.get_report()
    """
    
    # Default thresholds
    DEFAULT_THRESHOLDS = {
        # Z-score thresholds for distribution shift
        'z_score_warning': 2.5,
        'z_score_critical': 3.5,
        
        # Missing data thresholds (% of expected)
        'missing_warning_pct': 10,
        'missing_critical_pct': 25,
        
        # Staleness thresholds (hours)
        'stale_warning_hours': 4,
        'stale_critical_hours': 24,
        
        # Coverage thresholds (% of universe)
        'coverage_warning_pct': 80,
        'coverage_critical_pct': 60,
        
        # Outlier spike threshold (% of values)
        'outlier_spike_pct': 20,
    }
    
    # Metrics to track
    TRACKED_METRICS = [
        'sentiment_score',
        'fundamental_score', 
        'technical_score',
        'options_flow_score',
        'institutional_score',
        'total_score',
        'article_count',
        'confidence',
    ]
    
    def __init__(self, thresholds: Dict = None):
        """
        Initialize drift monitor.
        
        Args:
            thresholds: Custom thresholds (merged with defaults)
        """
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        
        # Metric statistics by name
        self.metrics: Dict[str, MetricStats] = {}
        
        # Coverage tracking by date
        self.daily_coverage: Dict[str, Dict[str, int]] = {}  # date -> {metric: count}
        
        # Alert history
        self.alerts: List[DriftAlert] = []
        self.alert_counter = 0
        
        # Expected universe size
        self.expected_universe_size = 100
        
        # Last check time
        self.last_check: Optional[datetime] = None
    
    def record_metric(self,
                      metric_name: str,
                      value: float,
                      ticker: str = None,
                      timestamp: datetime = None):
        """
        Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            ticker: Optional ticker for context
            timestamp: Optional timestamp (defaults to now)
        """
        timestamp = timestamp or datetime.now()
        
        # Initialize metric stats if needed
        if metric_name not in self.metrics:
            self.metrics[metric_name] = MetricStats(name=metric_name)
        
        # Update statistics
        self.metrics[metric_name].update(value, timestamp)
        
        # Update daily coverage
        date_str = timestamp.strftime('%Y-%m-%d')
        if date_str not in self.daily_coverage:
            self.daily_coverage[date_str] = {}
        
        if metric_name not in self.daily_coverage[date_str]:
            self.daily_coverage[date_str][metric_name] = 0
        self.daily_coverage[date_str][metric_name] += 1
    
    def record_batch(self, 
                     records: List[Dict],
                     timestamp: datetime = None):
        """
        Record metrics from a batch of scoring results.
        
        Args:
            records: List of dicts with metric values
            timestamp: Timestamp for all records
        """
        for record in records:
            ticker = record.get('ticker', '')
            for metric in self.TRACKED_METRICS:
                if metric in record and record[metric] is not None:
                    self.record_metric(metric, record[metric], ticker, timestamp)
    
    def check_drift(self) -> List[DriftAlert]:
        """
        Check for drift and anomalies.
        
        Returns:
            List of new alerts detected
        """
        new_alerts = []
        now = datetime.now()
        
        for metric_name, stats in self.metrics.items():
            # Check for stale data
            if stats.last_updated:
                hours_old = (now - stats.last_updated).total_seconds() / 3600
                
                if hours_old > self.thresholds['stale_critical_hours']:
                    new_alerts.append(self._create_alert(
                        DriftType.STALE_DATA,
                        AlertSeverity.CRITICAL,
                        metric_name,
                        hours_old,
                        (0, self.thresholds['stale_critical_hours']),
                        f"{metric_name} data is {hours_old:.1f} hours old"
                    ))
                elif hours_old > self.thresholds['stale_warning_hours']:
                    new_alerts.append(self._create_alert(
                        DriftType.STALE_DATA,
                        AlertSeverity.WARNING,
                        metric_name,
                        hours_old,
                        (0, self.thresholds['stale_warning_hours']),
                        f"{metric_name} data is {hours_old:.1f} hours old"
                    ))
            
            # Check for distribution shift (if enough data)
            if stats.count >= 50 and stats.recent_values:
                recent_mean = np.mean(list(stats.recent_values)[-20:])
                z_score = stats.get_z_score(recent_mean)
                
                if abs(z_score) > self.thresholds['z_score_critical']:
                    new_alerts.append(self._create_alert(
                        DriftType.DISTRIBUTION_SHIFT,
                        AlertSeverity.CRITICAL,
                        metric_name,
                        z_score,
                        (-self.thresholds['z_score_critical'], 
                         self.thresholds['z_score_critical']),
                        f"{metric_name} distribution shifted (z={z_score:.2f})"
                    ))
                elif abs(z_score) > self.thresholds['z_score_warning']:
                    new_alerts.append(self._create_alert(
                        DriftType.DISTRIBUTION_SHIFT,
                        AlertSeverity.WARNING,
                        metric_name,
                        z_score,
                        (-self.thresholds['z_score_warning'],
                         self.thresholds['z_score_warning']),
                        f"{metric_name} distribution drifting (z={z_score:.2f})"
                    ))
            
            # Check for outlier spike
            if stats.recent_values and len(stats.recent_values) >= 20:
                recent = list(stats.recent_values)[-20:]
                outlier_count = sum(
                    1 for v in recent 
                    if abs(stats.get_z_score(v)) > 3
                )
                outlier_pct = (outlier_count / len(recent)) * 100
                
                if outlier_pct > self.thresholds['outlier_spike_pct']:
                    new_alerts.append(self._create_alert(
                        DriftType.OUTLIER_SPIKE,
                        AlertSeverity.WARNING,
                        metric_name,
                        outlier_pct,
                        (0, self.thresholds['outlier_spike_pct']),
                        f"{metric_name} has {outlier_pct:.0f}% outliers recently"
                    ))
        
        # Check coverage
        today = datetime.now().strftime('%Y-%m-%d')
        if today in self.daily_coverage:
            for metric_name in self.TRACKED_METRICS:
                count = self.daily_coverage[today].get(metric_name, 0)
                coverage_pct = (count / self.expected_universe_size) * 100
                
                if coverage_pct < self.thresholds['coverage_critical_pct']:
                    new_alerts.append(self._create_alert(
                        DriftType.COVERAGE_DROP,
                        AlertSeverity.CRITICAL,
                        metric_name,
                        coverage_pct,
                        (self.thresholds['coverage_critical_pct'], 100),
                        f"{metric_name} coverage only {coverage_pct:.0f}% of universe"
                    ))
                elif coverage_pct < self.thresholds['coverage_warning_pct']:
                    new_alerts.append(self._create_alert(
                        DriftType.COVERAGE_DROP,
                        AlertSeverity.WARNING,
                        metric_name,
                        coverage_pct,
                        (self.thresholds['coverage_warning_pct'], 100),
                        f"{metric_name} coverage at {coverage_pct:.0f}% of universe"
                    ))
        
        # Store alerts and update check time
        self.alerts.extend(new_alerts)
        self.last_check = now
        
        return new_alerts
    
    def get_report(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring report.
        
        Returns:
            Dict with metric statistics, coverage, and alerts
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'metrics': {},
            'coverage': {},
            'recent_alerts': [],
            'health_status': 'healthy',
        }
        
        # Metric statistics
        for name, stats in self.metrics.items():
            report['metrics'][name] = {
                'count': stats.count,
                'mean': round(stats.mean, 2),
                'std': round(stats.std, 2),
                'min': round(stats.min_val, 2) if stats.min_val != float('inf') else None,
                'max': round(stats.max_val, 2) if stats.max_val != float('-inf') else None,
                'last_value': round(stats.last_value, 2),
                'last_updated': stats.last_updated.isoformat() if stats.last_updated else None,
            }
        
        # Today's coverage
        today = datetime.now().strftime('%Y-%m-%d')
        if today in self.daily_coverage:
            report['coverage'] = {
                metric: count 
                for metric, count in self.daily_coverage[today].items()
            }
        
        # Recent alerts (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        recent = [
            {
                'id': a.alert_id,
                'type': a.drift_type.value,
                'severity': a.severity.value,
                'metric': a.metric_name,
                'message': a.message,
                'detected_at': a.detected_at.isoformat(),
            }
            for a in self.alerts
            if a.detected_at >= cutoff
        ]
        report['recent_alerts'] = recent
        
        # Determine health status
        critical_count = sum(1 for a in recent if a['severity'] == 'critical')
        warning_count = sum(1 for a in recent if a['severity'] == 'warning')
        
        if critical_count > 0:
            report['health_status'] = 'critical'
        elif warning_count > 2:
            report['health_status'] = 'degraded'
        elif warning_count > 0:
            report['health_status'] = 'warning'
        
        return report
    
    def get_metric_history(self, 
                           metric_name: str,
                           last_n: int = 100) -> List[float]:
        """Get recent values for a metric."""
        if metric_name not in self.metrics:
            return []
        return list(self.metrics[metric_name].recent_values)[-last_n:]
    
    def set_expected_universe_size(self, size: int):
        """Set expected universe size for coverage calculations."""
        self.expected_universe_size = size
    
    def clear_old_alerts(self, hours: int = 24):
        """Clear alerts older than specified hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        self.alerts = [a for a in self.alerts if a.detected_at >= cutoff]
    
    def _create_alert(self,
                      drift_type: DriftType,
                      severity: AlertSeverity,
                      metric_name: str,
                      current_value: float,
                      expected_range: Tuple[float, float],
                      message: str) -> DriftAlert:
        """Create a new alert."""
        self.alert_counter += 1
        alert_id = f"DRIFT_{self.alert_counter:05d}"
        
        return DriftAlert(
            alert_id=alert_id,
            drift_type=drift_type,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            expected_range=expected_range,
            message=message,
        )


# =============================================================================
# DATA QUALITY DASHBOARD
# =============================================================================

class DataQualityDashboard:
    """
    Aggregates monitoring data into a dashboard view.
    
    Usage:
        dashboard = DataQualityDashboard(monitor)
        summary = dashboard.get_summary()
        print(dashboard.render_text())
    """
    
    def __init__(self, monitor: DriftMonitor):
        self.monitor = monitor
    
    def get_summary(self) -> Dict[str, Any]:
        """Get dashboard summary."""
        report = self.monitor.get_report()
        
        summary = {
            'health': report['health_status'],
            'metrics_tracked': len(report['metrics']),
            'alerts_24h': len(report['recent_alerts']),
            'critical_alerts': sum(
                1 for a in report['recent_alerts'] 
                if a['severity'] == 'critical'
            ),
            'warning_alerts': sum(
                1 for a in report['recent_alerts']
                if a['severity'] == 'warning'
            ),
        }
        
        # Find problematic metrics
        problem_metrics = []
        for alert in report['recent_alerts']:
            if alert['metric'] not in problem_metrics:
                problem_metrics.append(alert['metric'])
        summary['problem_metrics'] = problem_metrics[:5]
        
        return summary
    
    def render_text(self) -> str:
        """Render dashboard as text."""
        report = self.monitor.get_report()
        summary = self.get_summary()
        
        lines = [
            "=" * 60,
            "DATA QUALITY DASHBOARD",
            "=" * 60,
            f"Generated: {report['generated_at']}",
            "",
        ]
        
        # Health status with emoji
        health_emoji = {
            'healthy': '✅',
            'warning': '⚠️',
            'degraded': '🟡',
            'critical': '🔴',
        }
        lines.append(
            f"Health Status: {health_emoji.get(summary['health'], '❓')} "
            f"{summary['health'].upper()}"
        )
        lines.append("")
        
        # Alert summary
        lines.append(f"📊 Metrics Tracked: {summary['metrics_tracked']}")
        lines.append(f"🚨 Alerts (24h): {summary['alerts_24h']}")
        lines.append(f"   Critical: {summary['critical_alerts']}")
        lines.append(f"   Warning: {summary['warning_alerts']}")
        lines.append("")
        
        # Metric stats
        lines.append("📈 METRIC STATISTICS:")
        for name, stats in report['metrics'].items():
            lines.append(
                f"   {name}: mean={stats['mean']:.1f}, "
                f"std={stats['std']:.1f}, n={stats['count']}"
            )
        lines.append("")
        
        # Recent alerts
        if report['recent_alerts']:
            lines.append("🚨 RECENT ALERTS:")
            for alert in report['recent_alerts'][:5]:
                severity_icon = '🔴' if alert['severity'] == 'critical' else '⚠️'
                lines.append(f"   {severity_icon} {alert['message']}")
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_monitor_instance = None


def get_drift_monitor() -> DriftMonitor:
    """Get singleton drift monitor."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = DriftMonitor()
    return _monitor_instance


def record_metric(metric_name: str, value: float, **kwargs):
    """Quick access to record a metric."""
    get_drift_monitor().record_metric(metric_name, value, **kwargs)


def check_drift() -> List[DriftAlert]:
    """Quick access to check for drift."""
    return get_drift_monitor().check_drift()


def get_monitoring_report() -> Dict[str, Any]:
    """Quick access to get monitoring report."""
    return get_drift_monitor().get_report()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import random
    
    # Create monitor
    monitor = DriftMonitor()
    monitor.set_expected_universe_size(50)
    
    # Simulate some normal data
    print("Recording normal data...")
    for i in range(100):
        monitor.record_metric('sentiment_score', random.gauss(60, 15))
        monitor.record_metric('fundamental_score', random.gauss(55, 12))
        monitor.record_metric('technical_score', random.gauss(50, 18))
    
    # Check for drift (should be clean)
    alerts = monitor.check_drift()
    print(f"Initial alerts: {len(alerts)}")
    
    # Simulate a distribution shift
    print("\nSimulating distribution shift...")
    for i in range(30):
        monitor.record_metric('sentiment_score', random.gauss(85, 10))  # Shifted up
    
    # Check again
    alerts = monitor.check_drift()
    print(f"After shift alerts: {len(alerts)}")
    for alert in alerts:
        print(f"  {alert}")
    
    # Print dashboard
    dashboard = DataQualityDashboard(monitor)
    print("\n" + dashboard.render_text())
