"""Check actual schema and class names for dual_analyst.py fixes."""

# Check retrieval class name
print("=" * 50)
print("CHECKING RETRIEVAL CLASS")
print("=" * 50)
try:
    import src.rag.retrieval as retrieval_module
    classes = [name for name in dir(retrieval_module) if not name.startswith('_')]
    print(f"Available in src.rag.retrieval: {classes}")
except Exception as e:
    print(f"Error: {e}")

# Check screener_scores columns
print("\n" + "=" * 50)
print("CHECKING screener_scores TABLE")
print("=" * 50)
try:
    from src.db.connection import get_connection
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'screener_scores'
                ORDER BY ordinal_position
            """)
            cols = [row[0] for row in cur.fetchall()]
            print(f"Columns: {cols}")
except Exception as e:
    print(f"Error: {e}")

# Check fundamentals columns
print("\n" + "=" * 50)
print("CHECKING fundamentals TABLE")
print("=" * 50)
try:
    from src.db.connection import get_connection
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'fundamentals'
                ORDER BY ordinal_position
            """)
            cols = [row[0] for row in cur.fetchall()]
            print(f"Columns: {cols}")
except Exception as e:
    print(f"Error: {e}")

# Check prices columns
print("\n" + "=" * 50)
print("CHECKING prices TABLE")
print("=" * 50)
try:
    from src.db.connection import get_connection
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'prices'
                ORDER BY ordinal_position
            """)
            cols = [row[0] for row in cur.fetchall()]
            print(f"Columns: {cols}")
except Exception as e:
    print(f"Error: {e}")

# Check options_flow tables
print("\n" + "=" * 50)
print("CHECKING options flow TABLES")
print("=" * 50)
try:
    from src.db.connection import get_connection
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name LIKE '%option%'
            """)
            tables = [row[0] for row in cur.fetchall()]
            print(f"Options tables: {tables}")

            for table in tables[:3]:
                cur.execute(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = '{table}'
                    LIMIT 10
                """)
                cols = [row[0] for row in cur.fetchall()]
                print(f"  {table}: {cols}")
except Exception as e:
    print(f"Error: {e}")