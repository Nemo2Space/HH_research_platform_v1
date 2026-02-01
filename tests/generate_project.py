"""
Alpha Platform - Project File Generator

Run this script to create all project files.
Usage: python generate_project.py

This will create/overwrite files in the current directory.
"""

import os

# =============================================================================
# FILE DEFINITIONS
# =============================================================================

FILES = {
    # -------------------------------------------------------------------------
    # Source Package Init Files
    # -------------------------------------------------------------------------
    "src/__init__.py": '''"""Alpha Platform - Source Package"""
__version__ = "1.0.0"
''',

    "src/db/__init__.py": '''"""Alpha Platform - Database Package"""
from .connection import get_connection, get_engine, DatabaseConfig, test_connection

__all__ = ["get_connection", "get_engine", "DatabaseConfig", "test_connection"]
''',

    "src/kafka/__init__.py": '''"""Alpha Platform - Kafka Package"""
''',

    "src/data/__init__.py": '''"""Alpha Platform - Data Ingestion Package"""
''',

    "src/screener/__init__.py": '''"""Alpha Platform - Screener Package"""
''',

    "src/committee/__init__.py": '''"""Alpha Platform - Committee Package"""
''',

    "src/committee/agents/__init__.py": '''"""Alpha Platform - Agents Package"""
''',

    "src/llm/__init__.py": '''"""Alpha Platform - LLM Package"""
''',

    "src/portfolio/__init__.py": '''"""Alpha Platform - Portfolio Package"""
''',

    "src/utils/__init__.py": '''"""Alpha Platform - Utilities Package"""
from .logging import get_logger

__all__ = ["get_logger"]
''',

    # -------------------------------------------------------------------------
    # Database Module
    # -------------------------------------------------------------------------
    "src/db/connection.py": '''"""
Alpha Platform - Database Connection

Handles PostgreSQL/TimescaleDB connection management.
"""

import os
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "alpha_platform"
    user: str = "alpha"
    password: str = ""
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "alpha_platform"),
            user=os.getenv("POSTGRES_USER", "alpha"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )
    
    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def dsn(self) -> str:
        """Get DSN for psycopg2."""
        return f"host={self.host} port={self.port} dbname={self.database} user={self.user} password={self.password}"


# Global connection pool
_connection_pool: Optional[pool.ThreadedConnectionPool] = None


def init_connection_pool(config: Optional[DatabaseConfig] = None, min_conn: int = 1, max_conn: int = 10):
    """Initialize the connection pool."""
    global _connection_pool
    
    if config is None:
        config = DatabaseConfig.from_env()
    
    _connection_pool = pool.ThreadedConnectionPool(
        min_conn,
        max_conn,
        host=config.host,
        port=config.port,
        database=config.database,
        user=config.user,
        password=config.password,
    )
    return _connection_pool


def get_connection_pool() -> pool.ThreadedConnectionPool:
    """Get or create the connection pool."""
    global _connection_pool
    if _connection_pool is None:
        init_connection_pool()
    return _connection_pool


@contextmanager
def get_connection():
    """
    Get a database connection from the pool.
    
    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM prices LIMIT 10")
                rows = cur.fetchall()
    """
    pool_obj = get_connection_pool()
    conn = pool_obj.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool_obj.putconn(conn)


def get_engine(config: Optional[DatabaseConfig] = None) -> Engine:
    """
    Get SQLAlchemy engine for pandas operations.
    
    Usage:
        engine = get_engine()
        df = pd.read_sql("SELECT * FROM prices", engine)
    """
    if config is None:
        config = DatabaseConfig.from_env()
    
    return create_engine(
        config.connection_string,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )


def test_connection(config: Optional[DatabaseConfig] = None) -> bool:
    """
    Test database connection.
    
    Returns True if connection successful, raises exception otherwise.
    """
    if config is None:
        config = DatabaseConfig.from_env()
    
    conn = psycopg2.connect(
        host=config.host,
        port=config.port,
        database=config.database,
        user=config.user,
        password=config.password,
    )
    
    with conn.cursor() as cur:
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        print(f"Connected to: {version[:60]}...")
        
        cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
        table_count = cur.fetchone()[0]
        print(f"Tables found: {table_count}")
    
    conn.close()
    return True
''',

    # -------------------------------------------------------------------------
    # Utils Module
    # -------------------------------------------------------------------------
    "src/utils/logging.py": '''"""
Alpha Platform - Logging Configuration
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger.
    
    Args:
        name: Logger name (usually __name__)
        level: Log level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    if level is None:
        level = logging.INFO
    
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(level)
    return logger
''',

    # -------------------------------------------------------------------------
    # Test Scripts
    # -------------------------------------------------------------------------
    "scripts/test_db_connection.py": '''"""
Test Database Connection

Run this script to verify the database connection works.

Usage:
    python scripts/test_db_connection.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

# Load .env file
load_dotenv()

from src.db.connection import test_connection, DatabaseConfig


def main():
    print("=" * 50)
    print("Alpha Platform - Database Connection Test")
    print("=" * 50)
    print()
    
    # Show config (hide password)
    config = DatabaseConfig.from_env()
    print(f"Host:     {config.host}")
    print(f"Port:     {config.port}")
    print(f"Database: {config.database}")
    print(f"User:     {config.user}")
    print(f"Password: {'*' * len(config.password) if config.password else '(not set)'}")
    print()
    
    if not config.password:
        print("ERROR: POSTGRES_PASSWORD not set!")
        print("Please set it in your .env file")
        sys.exit(1)
    
    try:
        test_connection(config)
        print()
        print("=" * 50)
        print("SUCCESS - Database connection working!")
        print("=" * 50)
    except Exception as e:
        print()
        print("=" * 50)
        print(f"FAILED - {e}")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()
''',

    # -------------------------------------------------------------------------
    # Dashboard Files
    # -------------------------------------------------------------------------
    "dashboard/__init__.py": '''"""Alpha Platform - Dashboard Package"""
''',

    "dashboard/components/__init__.py": '''"""Dashboard Components"""
''',

    "dashboard/pages/__init__.py": '''"""Dashboard Pages"""
''',

    # -------------------------------------------------------------------------
    # Workers
    # -------------------------------------------------------------------------
    "workers/__init__.py": '''"""Alpha Platform - Workers"""
''',

    # -------------------------------------------------------------------------
    # Tests
    # -------------------------------------------------------------------------
    "tests/__init__.py": '''"""Alpha Platform - Tests"""
''',

}


# =============================================================================
# GENERATOR FUNCTION
# =============================================================================

def generate_files():
    """Generate all project files."""
    print("=" * 60)
    print("Alpha Platform - Project File Generator")
    print("=" * 60)
    print()

    created = 0
    updated = 0

    for filepath, content in FILES.items():
        # Create directory if needed
        dir_path = os.path.dirname(filepath)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}/")

        # Check if file exists
        exists = os.path.exists(filepath)

        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        if exists:
            print(f"Updated: {filepath}")
            updated += 1
        else:
            print(f"Created: {filepath}")
            created += 1

    print()
    print("=" * 60)
    print(f"Done! Created: {created}, Updated: {updated}")
    print("=" * 60)


if __name__ == "__main__":
    generate_files()