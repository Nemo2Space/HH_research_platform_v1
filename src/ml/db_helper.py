"""
Database Helper for ML Module

Provides database connections using credentials from .env or hardcoded fallback.

Location: src/ml/db_helper.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to load .env
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Database configuration
DB_CONFIG = {
    'host': os.environ.get('POSTGRES_HOST', 'localhost'),
    'port': os.environ.get('POSTGRES_PORT', '5432'),
    'database': os.environ.get('POSTGRES_DB', 'alpha_platform'),
    'user': os.environ.get('POSTGRES_USER', 'alpha'),
    'password': os.environ.get('POSTGRES_PASSWORD', 'alpha_secure_2024'),
}


def get_connection_string():
    """Get PostgreSQL connection string."""
    return (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )


def get_engine():
    """Get SQLAlchemy engine."""
    # Try to use existing connection module first
    try:
        from src.db.connection import get_engine as existing_engine
        return existing_engine()
    except ImportError:
        pass

    # Fallback: create engine with credentials
    from sqlalchemy import create_engine
    return create_engine(get_connection_string())


def get_connection():
    """Get psycopg2 connection."""
    # Try to use existing connection module first
    try:
        from src.db.connection import get_connection as existing_conn
        return existing_conn()
    except ImportError:
        pass

    # Fallback: create connection with credentials
    import psycopg2
    return psycopg2.connect(
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        dbname=DB_CONFIG['database'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password']
    )


# Test connection
if __name__ == "__main__":
    print("Testing database connection...")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        print("✅ Database connection successful!")
        conn.close()
    except Exception as e:
        print(f"❌ Connection failed: {e}")