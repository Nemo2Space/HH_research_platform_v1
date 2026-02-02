"""
Alert Scheduler

Runs scheduled alerts at specified times:
- Daily morning summary (configurable time)
- Periodic signal checks
- Earnings reminders

Uses APScheduler for reliable scheduling.

Author: Alpha Research Platform
"""

import os
import sys
from datetime import datetime, time
from typing import Callable, List, Optional
import threading

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import APScheduler
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    logger.warning("APScheduler not installed. Run: pip install apscheduler")


class AlertScheduler:
    """
    Schedules and runs automated alerts.
    """

    def __init__(self, timezone: str = "Europe/Zurich"):
        """
        Initialize scheduler.

        Args:
            timezone: Timezone for scheduling (default: Europe/Zurich)
        """
        self.timezone = timezone
        self.scheduler = None
        self.running = False

        if SCHEDULER_AVAILABLE:
            self.scheduler = BackgroundScheduler(timezone=timezone)
            logger.info(f"Alert scheduler initialized (timezone: {timezone})")
        else:
            logger.error("APScheduler not available - install with: pip install apscheduler")

    def start(self):
        """Start the scheduler."""
        if self.scheduler and not self.running:
            self.scheduler.start()
            self.running = True
            logger.info("Alert scheduler started")

    def stop(self):
        """Stop the scheduler."""
        if self.scheduler and self.running:
            self.scheduler.shutdown()
            self.running = False
            logger.info("Alert scheduler stopped")

    def add_daily_summary(self, hour: int = 14, minute: int = 40):
        """
        Schedule daily summary alert.

        Args:
            hour: Hour to run (24h format)
            minute: Minute to run
        """
        if not self.scheduler:
            logger.error("Scheduler not available")
            return

        def job():
            logger.info("Running scheduled daily summary...")
            try:
                from src.alerts.telegram_alerts import send_daily_summary
                send_daily_summary()
                logger.info("Daily summary sent successfully")
            except Exception as e:
                logger.error(f"Error sending daily summary: {e}")

        self.scheduler.add_job(
            job,
            CronTrigger(hour=hour, minute=minute),
            id='daily_summary',
            name='Daily Market Summary',
            replace_existing=True
        )

        logger.info(f"Scheduled daily summary at {hour:02d}:{minute:02d} {self.timezone}")

    def add_signal_check(self, interval_minutes: int = 60):
        """
        Schedule periodic signal change checks.

        Args:
            interval_minutes: Check interval in minutes
        """
        if not self.scheduler:
            return

        def job():
            logger.info("Running scheduled signal check...")
            try:
                from src.alerts.telegram_alerts import get_alerter
                alerter = get_alerter()
                alerts = alerter.check_signal_changes()
                for alert in alerts:
                    alerter.send_alert(alert)
                if alerts:
                    logger.info(f"Sent {len(alerts)} signal change alerts")
            except Exception as e:
                logger.error(f"Error checking signals: {e}")

        self.scheduler.add_job(
            job,
            IntervalTrigger(minutes=interval_minutes),
            id='signal_check',
            name='Signal Change Check',
            replace_existing=True
        )

        logger.info(f"Scheduled signal check every {interval_minutes} minutes")

    def add_regime_check(self, interval_minutes: int = 120):
        """
        Schedule periodic regime change checks.

        Args:
            interval_minutes: Check interval in minutes
        """
        if not self.scheduler:
            return

        def job():
            logger.info("Running scheduled regime check...")
            try:
                from src.alerts.telegram_alerts import get_alerter
                alerter = get_alerter()
                alert = alerter.check_regime_change()
                if alert:
                    alerter.send_alert(alert)
                    logger.info("Sent regime change alert")
            except Exception as e:
                logger.error(f"Error checking regime: {e}")

        self.scheduler.add_job(
            job,
            IntervalTrigger(minutes=interval_minutes),
            id='regime_check',
            name='Regime Change Check',
            replace_existing=True
        )

        logger.info(f"Scheduled regime check every {interval_minutes} minutes")

    def add_earnings_reminder(self, hour: int = 8, minute: int = 0):
        """
        Schedule daily earnings reminder (morning).

        Args:
            hour: Hour to run
            minute: Minute to run
        """
        if not self.scheduler:
            return

        def job():
            logger.info("Running scheduled earnings check...")
            try:
                from src.alerts.telegram_alerts import get_alerter
                alerter = get_alerter()
                alerts = alerter.check_upcoming_earnings(days_ahead=3)
                for alert in alerts:
                    alerter.send_alert(alert)
                if alerts:
                    logger.info(f"Sent {len(alerts)} earnings alerts")
            except Exception as e:
                logger.error(f"Error checking earnings: {e}")

        self.scheduler.add_job(
            job,
            CronTrigger(hour=hour, minute=minute),
            id='earnings_reminder',
            name='Earnings Reminder',
            replace_existing=True
        )

        logger.info(f"Scheduled earnings reminder at {hour:02d}:{minute:02d} {self.timezone}")

    def add_bond_check(self, hour: int = 15, minute: int = 0):
        """
        Schedule daily bond opportunity check.

        Args:
            hour: Hour to run
            minute: Minute to run
        """
        if not self.scheduler:
            return

        def job():
            logger.info("Running scheduled bond check...")
            try:
                from src.alerts.telegram_alerts import get_alerter
                alerter = get_alerter()
                alerts = alerter.check_bond_opportunities()
                for alert in alerts:
                    alerter.send_alert(alert)
                if alerts:
                    logger.info(f"Sent {len(alerts)} bond alerts")
            except Exception as e:
                logger.error(f"Error checking bonds: {e}")

        self.scheduler.add_job(
            job,
            CronTrigger(hour=hour, minute=minute),
            id='bond_check',
            name='Bond Opportunity Check',
            replace_existing=True
        )

        logger.info(f"Scheduled bond check at {hour:02d}:{minute:02d} {self.timezone}")

    def add_custom_job(self, job_id: str, func: Callable,
                       hour: int = None, minute: int = None,
                       interval_minutes: int = None):
        """
        Add a custom scheduled job.

        Args:
            job_id: Unique job identifier
            func: Function to run
            hour: Hour for daily job (if set)
            minute: Minute for daily job
            interval_minutes: Interval for periodic job (if set)
        """
        if not self.scheduler:
            return

        if hour is not None:
            trigger = CronTrigger(hour=hour, minute=minute or 0)
        elif interval_minutes:
            trigger = IntervalTrigger(minutes=interval_minutes)
        else:
            logger.error("Must specify either hour or interval_minutes")
            return

        self.scheduler.add_job(
            func,
            trigger,
            id=job_id,
            replace_existing=True
        )

        logger.info(f"Added custom job: {job_id}")

    def list_jobs(self) -> List[dict]:
        """List all scheduled jobs."""
        if not self.scheduler:
            return []

        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': str(job.next_run_time),
                'trigger': str(job.trigger)
            })

        return jobs

    def remove_job(self, job_id: str):
        """Remove a scheduled job."""
        if self.scheduler:
            try:
                self.scheduler.remove_job(job_id)
                logger.info(f"Removed job: {job_id}")
            except Exception as e:
                logger.error(f"Error removing job {job_id}: {e}")

    def setup_default_schedule(self):
        """
        Set up default alert schedule.

        Default schedule (Zurich time):
        - 14:40 - Daily summary
        - 08:00 - Earnings reminder
        - 15:00 - Bond check
        - Every 2 hours - Regime check
        """
        self.add_daily_summary(hour=14, minute=40)
        self.add_earnings_reminder(hour=8, minute=0)
        self.add_bond_check(hour=15, minute=0)
        self.add_regime_check(interval_minutes=120)

        # ML auto-maintenance: daily return backfill + weekly model retrain
        try:
            from src.ml.auto_maintenance import register_ml_jobs
            register_ml_jobs(self)
        except Exception as e:
            logger.warning(f"Could not register ML maintenance jobs: {e}")

        logger.info("Default schedule configured")


# ============================================================
# Global Scheduler Instance
# ============================================================

_scheduler = None


def get_scheduler() -> AlertScheduler:
    """Get singleton scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = AlertScheduler(timezone="Europe/Zurich")
    return _scheduler


def start_scheduler():
    """Start the global scheduler with default settings."""
    scheduler = get_scheduler()
    scheduler.setup_default_schedule()
    scheduler.start()
    return scheduler


def stop_scheduler():
    """Stop the global scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.stop()


# ============================================================
# Standalone Runner Script
# ============================================================

def run_scheduler_standalone():
    """
    Run scheduler as standalone process.

    Usage:
        python -m src.alerts.scheduler
    """
    import signal

    print("=" * 60)
    print("🕐 Alpha Research Alert Scheduler")
    print("=" * 60)
    print(f"Timezone: Europe/Zurich")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    scheduler = start_scheduler()

    # Print scheduled jobs
    print("📋 Scheduled Jobs:")
    for job in scheduler.list_jobs():
        print(f"  - {job['name']}: {job['next_run']}")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)

    # Handle shutdown gracefully
    def shutdown(signum, frame):
        print("\n\nShutting down scheduler...")
        stop_scheduler()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep running
    try:
        while True:
            import time
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        shutdown(None, None)


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run_scheduler_standalone()
    else:
        print("Alert Scheduler Test\n")

        if not SCHEDULER_AVAILABLE:
            print("❌ APScheduler not installed")
            print("Run: pip install apscheduler")
            sys.exit(1)

        scheduler = AlertScheduler(timezone="Europe/Zurich")
        scheduler.setup_default_schedule()

        print("📋 Scheduled Jobs:")
        for job in scheduler.list_jobs():
            print(f"  - {job['name']}")
            print(f"    Next run: {job['next_run']}")
            print()

        print("\nTo run the scheduler:")
        print("  python -m src.alerts.scheduler run")
        print("\nOr add to your Streamlit app:")
        print("  from src.alerts.scheduler import start_scheduler")
        print("  start_scheduler()")