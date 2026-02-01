"""
Alpha Platform - Database Connection

Handles PostgreSQL/TimescaleDB connection management.
"""

import os
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager

# Load .env file BEFORE accessing environment variables
from dotenv import load_dotenv
load_dotenv()

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