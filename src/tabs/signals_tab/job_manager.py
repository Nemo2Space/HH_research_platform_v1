"""
Signals Tab - Job Manager

Background job tracking and management.
"""

from .shared import (
    st, pd, logger, _to_native, text,
    get_engine, SIGNAL_HUB_AVAILABLE, DB_AVAILABLE,
)

if SIGNAL_HUB_AVAILABLE:
    from .shared import get_signal_engine

def _get_job_status() -> dict:
    """Get job status from database."""
    try:
        engine = get_engine()
        df = pd.read_sql("SELECT * FROM analysis_job ORDER BY id DESC LIMIT 1", engine)
        if df.empty:
            return {'status': 'idle', 'processed_count': 0, 'total_count': 0}
        row = df.iloc[0]
        return {
            'status': row.get('status', 'idle'),
            'processed_count': int(row.get('processed_count', 0) or 0),
            'total_count': int(row.get('total_count', 0) or 0),
        }
    except:
        return {'status': 'idle', 'processed_count': 0, 'total_count': 0}


def _ensure_job_tables():
    """Create job tables if needed."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Create main job tracking table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS analysis_job (
                    id SERIAL PRIMARY KEY,
                    status VARCHAR(20) DEFAULT 'idle',
                    total_count INTEGER DEFAULT 0,
                    processed_count INTEGER DEFAULT 0,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            # Create log table with UNIQUE constraint on ticker
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS analysis_job_log (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) UNIQUE,
                    status VARCHAR(10),
                    news_count INTEGER DEFAULT 0,
                    sentiment_score INTEGER,
                    options_score INTEGER,
                    processed_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            # Initialize job row if needed
            result = conn.execute(text("SELECT COUNT(*) FROM analysis_job"))
            if result.fetchone()[0] == 0:
                conn.execute(text("INSERT INTO analysis_job (status) VALUES ('idle')"))
            conn.commit()
    except Exception as e:
        logger.debug(f"Init job tables error: {e}")


def _start_job(total: int):
    """Start or resume job."""
    from datetime import date as dt_date

    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Check if we have existing progress AND if it's from today
            result = conn.execute(text("""
                SELECT processed_count, total_count, status, DATE(updated_at) as last_date 
                FROM analysis_job LIMIT 1
            """))
            row = result.fetchone()

            today = dt_date.today()

            # Resume only if: has progress AND same day AND status allows resume
            if row and row[0] > 0 and row[3] == today and row[2] in ('stopped', 'idle', 'completed'):
                # Resume same-day job - keep progress, update total (in case universe changed)
                conn.execute(text(
                    "UPDATE analysis_job SET status='running', total_count=:t, updated_at=NOW()"),
                    {'t': total})
                logger.info(f"Resuming today's job: {row[0]} already processed, {total} total tickers")
            else:
                # Fresh start (new day or first run) - clear the log
                if row and row[3] != today:
                    logger.info(f"New day detected (last run: {row[3]}, today: {today}) - starting fresh")
                conn.execute(text("DROP TABLE IF EXISTS analysis_job_log"))
                conn.execute(text("""
                    CREATE TABLE analysis_job_log (
                        id SERIAL PRIMARY KEY,
                        ticker VARCHAR(10) UNIQUE,
                        status VARCHAR(10),
                        news_count INTEGER DEFAULT 0,
                        sentiment_score INTEGER,
                        options_score INTEGER,
                        processed_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """))
                conn.execute(text(
                    "UPDATE analysis_job SET status='running', total_count=:t, processed_count=0, updated_at=NOW()"),
                    {'t': total})
                logger.info(f"Fresh start: {total} tickers")
            conn.commit()
    except Exception as e:
        st.error(f"Start error: {e}")

def _stop_job():
    """Stop job."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("UPDATE analysis_job SET status='stopped', updated_at=NOW()"))
            conn.commit()
    except:
        pass


def _complete_job():
    """Mark job complete and clear all caches for fresh data."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("UPDATE analysis_job SET status='completed', updated_at=NOW()"))
            conn.commit()

        # Clear all caches so Load Signals shows fresh data
        keys_to_clear = ['signals_data', 'market_overview']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        # Clear SignalEngine cache entirely
        try:
            engine = get_signal_engine()
            if hasattr(engine, '_cache'):
                engine._cache.clear()
                logger.info("SignalEngine cache cleared")
        except:
            pass

        # Set force refresh flag and trigger signals load
        st.session_state.force_refresh = True
        st.session_state.signals_loaded = True
        st.session_state.show_analysis = False  # Switch to signals view
        logger.info("Analysis job completed - all caches cleared, switching to signals view")
    except Exception as e:
        logger.error(f"Complete job error: {e}")


def _reset_job():
    """Reset job - drop and recreate log table."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS analysis_job_log"))
            conn.execute(text("""
                CREATE TABLE analysis_job_log (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) UNIQUE,
                    status VARCHAR(10),
                    news_count INTEGER DEFAULT 0,
                    sentiment_score INTEGER,
                    options_score INTEGER,
                    processed_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            conn.execute(text("UPDATE analysis_job SET status='idle', processed_count=0, total_count=0"))
            conn.commit()
    except Exception as e:
        logger.warning(f"Reset job error: {e}")


def _log_result(ticker: str, status: str, news: int, sentiment, options,
                fundamental=None, technical=None, committee=None):
    """Log or update ticker result. Only increment processed_count for NEW entries."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Check if ticker already in log
            result = conn.execute(text("SELECT 1 FROM analysis_job_log WHERE ticker = :t"), {'t': ticker})
            exists = result.fetchone() is not None

            if exists:
                # Update existing entry - DO NOT increment processed_count
                conn.execute(text("""
                    UPDATE analysis_job_log SET
                        status = :s,
                        news_count = :n,
                        sentiment_score = :sent,
                        options_score = :opt,
                        processed_at = NOW()
                    WHERE ticker = :t
                """), {'t': ticker, 's': status, 'n': news, 'sent': _to_native(sentiment), 'opt': _to_native(options)})
            else:
                # Insert new entry AND increment processed_count
                conn.execute(text("""
                    INSERT INTO analysis_job_log (ticker, status, news_count, sentiment_score, options_score)
                    VALUES (:t, :s, :n, :sent, :opt)
                """), {'t': ticker, 's': status, 'n': news, 'sent': _to_native(sentiment), 'opt': _to_native(options)})
                conn.execute(text("UPDATE analysis_job SET processed_count=processed_count+1, updated_at=NOW()"))

            conn.commit()
        logger.info(f"{ticker}: {status} | News:{news} Sent:{sentiment} Opts:{options} Fund:{fundamental} Tech:{technical} Committee:{committee}")
    except Exception as e:
        logger.warning(f"Log result error for {ticker}: {e}")


def _get_processed_tickers() -> list:
    """Get already processed tickers."""
    try:
        engine = get_engine()
        df = pd.read_sql("SELECT ticker FROM analysis_job_log", engine)
        return df['ticker'].tolist() if not df.empty else []
    except:
        return []


def _get_job_log() -> pd.DataFrame:
    """Get job log."""
    try:
        engine = get_engine()
        return pd.read_sql("""
                           SELECT ticker,
                                  status,
                                  news_count,
                                  sentiment_score,
                                  options_score,
                                  TO_CHAR(processed_at, 'HH24:MI:SS') as time
                           FROM analysis_job_log
                           ORDER BY processed_at DESC LIMIT 30
                           """, engine)
    except:
        return pd.DataFrame()


def _get_job_stats() -> dict:
    """Get FULL job statistics (not limited to 30 like the log display)."""
    try:
        engine = get_engine()
        df = pd.read_sql("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE status = '✅') as analyzed,
                COUNT(*) FILTER (WHERE status = '⏭️') as skipped,
                COUNT(*) FILTER (WHERE status = '⚠️') as failed,
                COALESCE(SUM(news_count), 0) as news_total
            FROM analysis_job_log
        """, engine)
        if df.empty:
            return {'total': 0, 'analyzed': 0, 'skipped': 0, 'failed': 0, 'news_total': 0}
        row = df.iloc[0]
        return {
            'total': int(row['total'] or 0),
            'analyzed': int(row['analyzed'] or 0),
            'skipped': int(row['skipped'] or 0),
            'failed': int(row['failed'] or 0),
            'news_total': int(row['news_total'] or 0)
        }
    except Exception as e:
        logger.debug(f"Stats query error: {e}")
        return {'total': 0, 'analyzed': 0, 'skipped': 0, 'failed': 0, 'news_total': 0}