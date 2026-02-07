import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from src.ml.auto_maintenance import backfill_returns_job
backfill_returns_job()
