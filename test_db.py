from src.db.connection import get_engine
from sqlalchemy import text
with get_engine().connect() as conn:
    conn.execute(text("UPDATE analysis_job SET status='idle', processed_count=0, total_count=0"))
    conn.execute(text("DELETE FROM analysis_job_log"))
    conn.commit()
    print("Job reset. Log cleared.")
