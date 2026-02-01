import streamlit as st
import pandas as pd
import psycopg2
import json
import yfinance as yf
from datetime import datetime

def get_db_connection():
    return psycopg2.connect(
        host='localhost', 
        port=5432, 
        dbname='alpha_platform', 
        user='alpha', 
        password='alpha_secure_2024'
    )

def get_stock_info(tickers: list) -> pd.DataFrame:
    """Fetch stock info from database"""
    conn = get_db_connection()
    
    placeholders = ','.join(['%s'] * len(tickers))
    query = f'''
        SELECT DISTINCT ON (f.ticker)
            f.ticker,
            f.sector,
            f.market_cap,
            s.fundamental_score,
            s.sentiment_score,
            s.growth_score,
            t.signal_type as signal,
            a.ai_probability
        FROM fundamentals f
        LEFT JOIN screener_scores s ON f.ticker = s.ticker 
            AND s.date = (SELECT MAX(date) FROM screener_scores WHERE ticker = f.ticker)
        LEFT JOIN trading_signals t ON f.ticker = t.ticker 
            AND t.date = (SELECT MAX(date) FROM trading_signals WHERE ticker = f.ticker)
        LEFT JOIN ai_recommendations a ON f.ticker = a.ticker 
            AND a.recommendation_date = (SELECT MAX(recommendation_date) FROM ai_recommendations WHERE ticker = f.ticker)
        WHERE f.ticker IN ({placeholders})
        ORDER BY f.ticker, f.date DESC
    '''
    
    df = pd.read_sql(query, conn, params=tickers)
    conn.close()
    return df

def check_missing_tickers(tickers: list) -> tuple:
    """Check which tickers are missing from database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT DISTINCT ticker FROM fundamentals WHERE ticker IN %s", (tuple(tickers),))
    found = [row[0] for row in cur.fetchall()]
    conn.close()
    
    missing = [t for t in tickers if t.upper() not in [f.upper() for f in found]]
    return found, missing

def add_ticker_to_db(ticker: str) -> dict:
    """Add a new ticker to database using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info.get('symbol'):
            return {'success': False, 'error': 'Ticker not found on Yahoo Finance'}
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('''
            INSERT INTO fundamentals (ticker, sector, market_cap, pe_ratio, dividend_yield, date)
            VALUES (%s, %s, %s, %s, %s, CURRENT_DATE)
            ON CONFLICT (ticker, date) DO UPDATE SET
                sector = EXCLUDED.sector,
                market_cap = EXCLUDED.market_cap,
                pe_ratio = EXCLUDED.pe_ratio,
                dividend_yield = EXCLUDED.dividend_yield
        ''', (
            ticker.upper(),
            info.get('sector', 'Unknown'),
            info.get('marketCap'),
            info.get('trailingPE'),
            info.get('dividendYield')
        ))
        
        conn.commit()
        conn.close()
        
        return {
            'success': True, 
            'name': info.get('shortName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'market_cap': info.get('marketCap')
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def format_market_cap(value):
    """Format market cap for display"""
    if pd.isna(value) or value is None:
        return "N/A"
    if value >= 1e12:
        return f"\T"
    if value >= 1e9:
        return f"\B"
    if value >= 1e6:
        return f"\M"
    return f"\"

# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(page_title="Manual Portfolio Builder", layout="wide")
st.title("🛠️ Manual Portfolio Builder")

# Initialize session state
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = None
if 'missing_tickers' not in st.session_state:
    st.session_state.missing_tickers = []

# ============================================================
# TAB 1: Manual Entry
# ============================================================
tab1, tab2, tab3 = st.tabs(["📝 Manual Entry", "📁 Import JSON", "💾 Export Portfolio"])

with tab1:
    st.subheader("Enter Tickers")
    
    ticker_input = st.text_area(
        "Enter tickers (comma-separated)",
        placeholder="NVDA, AMD, MSFT, GOOGL, META, AMZN, PLTR, CRWD...",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        load_btn = st.button("🔍 Load Stocks", type="primary")
    
    if load_btn and ticker_input:
        # Parse tickers
        tickers = [t.strip().upper() for t in ticker_input.replace('\n', ',').split(',') if t.strip()]
        tickers = list(set(tickers))  # Remove duplicates
        
        st.info(f"Processing {len(tickers)} tickers...")
        
        # Check for missing tickers
        found, missing = check_missing_tickers(tickers)
        
        if missing:
            st.session_state.missing_tickers = missing
            st.warning(f"⚠️ {len(missing)} tickers not in database: {', '.join(missing)}")
        
        if found:
            # Get stock info
            df = get_stock_info(found)
            
            # Add weight column (equal weight by default)
            equal_weight = round(100 / len(df), 2)
            df['weight_pct'] = equal_weight
            
            # Format market cap
            df['market_cap_fmt'] = df['market_cap'].apply(format_market_cap)
            
            st.session_state.portfolio_df = df
            st.success(f"✅ Loaded {len(df)} stocks")
    
    # Show missing tickers with option to add
    if st.session_state.missing_tickers:
        st.subheader("⚠️ Missing Tickers")
        
        missing_df = pd.DataFrame({'ticker': st.session_state.missing_tickers})
        st.dataframe(missing_df, width='stretch')
        
        if st.button("➕ Add Missing Tickers to Database"):
            progress = st.progress(0)
            results = []
            
            for i, ticker in enumerate(st.session_state.missing_tickers):
                result = add_ticker_to_db(ticker)
                result['ticker'] = ticker
                results.append(result)
                progress.progress((i + 1) / len(st.session_state.missing_tickers))
            
            # Show results
            added = [r for r in results if r['success']]
            failed = [r for r in results if not r['success']]
            
            if added:
                st.success(f"✅ Added {len(added)} tickers: {', '.join([r['ticker'] for r in added])}")
            if failed:
                st.error(f"❌ Failed to add: {', '.join([f\"{r['ticker']} ({r['error']})\" for r in failed])}")
            
            st.session_state.missing_tickers = [r['ticker'] for r in failed]
            st.rerun()
    
    # Show editable portfolio table
    if st.session_state.portfolio_df is not None:
        st.subheader("📊 Portfolio Holdings")
        
        df = st.session_state.portfolio_df.copy()
        
        # Editable dataframe
        edited_df = st.data_editor(
            df[['ticker', 'sector', 'market_cap_fmt', 'fundamental_score', 'sentiment_score', 'signal', 'ai_probability', 'weight_pct']],
            column_config={
                'ticker': st.column_config.TextColumn('Ticker', disabled=True),
                'sector': st.column_config.TextColumn('Sector', disabled=True),
                'market_cap_fmt': st.column_config.TextColumn('Market Cap', disabled=True),
                'fundamental_score': st.column_config.NumberColumn('Fund Score', disabled=True, format="%.0f"),
                'sentiment_score': st.column_config.NumberColumn('Sentiment', disabled=True, format="%.0f"),
                'signal': st.column_config.TextColumn('Signal', disabled=True),
                'ai_probability': st.column_config.NumberColumn('AI Prob', disabled=True, format="%.1f%%"),
                'weight_pct': st.column_config.NumberColumn('Weight %', min_value=0, max_value=100, step=0.1, format="%.1f%%"),
            },
            width='stretch',
            num_rows="fixed"
        )
        
        # Update weights in session state
        st.session_state.portfolio_df['weight_pct'] = edited_df['weight_pct']
        
        # Weight summary
        total_weight = edited_df['weight_pct'].sum()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Weight", f"{total_weight:.1f}%")
        with col2:
            st.metric("Holdings", len(edited_df))
        with col3:
            if abs(total_weight - 100) < 0.1:
                st.success("✅ Weights balanced")
            else:
                st.warning(f"⚠️ {100 - total_weight:.1f}% unallocated")
        
        # Quick weight buttons
        st.subheader("Quick Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⚖️ Equal Weight"):
                equal_weight = round(100 / len(df), 2)
                st.session_state.portfolio_df['weight_pct'] = equal_weight
                st.rerun()
        with col2:
            if st.button("📈 Score-Based Weight"):
                scores = df['fundamental_score'].fillna(50)
                total_score = scores.sum()
                st.session_state.portfolio_df['weight_pct'] = round(scores / total_score * 100, 2)
                st.rerun()
        with col3:
            if st.button("🔄 Market Cap Weight"):
                caps = df['market_cap'].fillna(df['market_cap'].median())
                total_cap = caps.sum()
                st.session_state.portfolio_df['weight_pct'] = round(caps / total_cap * 100, 2)
                st.rerun()

# ============================================================
# TAB 2: Import JSON
# ============================================================
with tab2:
    st.subheader("Import Portfolio from JSON")
    
    st.markdown('''
    **Expected JSON format:**
`json
    [
        {"ticker": "NVDA", "name": "NVIDIA", "weight": 10.0},
        {"ticker": "AMD", "name": "AMD Inc", "weight": 8.0},
        ...
    ]
`
    ''')
    
    uploaded_file = st.file_uploader("Upload JSON file", type=['json'])
    
    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            
            if isinstance(data, list) and len(data) > 0:
                # Parse JSON
                tickers = [item.get('ticker', '').upper() for item in data if item.get('ticker')]
                weights = {item.get('ticker', '').upper(): item.get('weight', 0) for item in data}
                
                st.info(f"Found {len(tickers)} tickers in JSON")
                
                # Check database
                found, missing = check_missing_tickers(tickers)
                
                if missing:
                    st.warning(f"⚠️ Missing from database: {', '.join(missing)}")
                    
                    if st.button("➕ Add Missing Tickers"):
                        for ticker in missing:
                            result = add_ticker_to_db(ticker)
                            if result['success']:
                                st.success(f"Added {ticker}")
                            else:
                                st.error(f"Failed to add {ticker}: {result['error']}")
                        st.rerun()
                
                if found:
                    # Load portfolio
                    df = get_stock_info(found)
                    df['weight_pct'] = df['ticker'].apply(lambda t: weights.get(t.upper(), 0))
                    df['market_cap_fmt'] = df['market_cap'].apply(format_market_cap)
                    
                    st.session_state.portfolio_df = df
                    st.success(f"✅ Loaded {len(df)} stocks from JSON")
                    
                    # Show preview
                    st.dataframe(df[['ticker', 'sector', 'market_cap_fmt', 'weight_pct']])
            else:
                st.error("Invalid JSON format - expected array of objects")
        except json.JSONDecodeError as e:
            st.error(f"JSON parse error: {e}")

# ============================================================
# TAB 3: Export Portfolio
# ============================================================
with tab3:
    st.subheader("Export Portfolio")
    
    if st.session_state.portfolio_df is not None:
        df = st.session_state.portfolio_df
        
        # JSON export
        export_data = df[['ticker', 'sector', 'weight_pct']].to_dict(orient='records')
        json_str = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="📥 Download as JSON",
            data=json_str,
            file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # CSV export
        csv_str = df[['ticker', 'sector', 'market_cap', 'weight_pct']].to_csv(index=False)
        
        st.download_button(
            label="📥 Download as CSV",
            data=csv_str,
            file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.subheader("Preview")
        st.json(export_data[:5])
    else:
        st.info("No portfolio loaded. Go to 'Manual Entry' or 'Import JSON' tab first.")
