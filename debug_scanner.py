import streamlit as st
import time
import logging

from src.analytics import TechnicalAnalyzer

logging.basicConfig(level=logging.INFO)

st.title("Full Scanner Path Test")

ticker = st.text_input("Ticker", value="PCAR")

if st.button("Run FULL _process_next equivalent"):
    
    st.write("**STEP 1: News...**")
    start = time.time()
    from src.data.news import NewsCollector
    nc = NewsCollector()
    articles = nc.collect_all_news(ticker)
    st.write(f"  ✅ {len(articles)} articles in {time.time()-start:.1f}s")

    st.write("**STEP 2: Sentiment...**")
    start = time.time()
    from src.screener.sentiment import SentimentAnalyzer
    sa = SentimentAnalyzer()
    sentiment_data = sa.analyze_ticker_sentiment(ticker, articles) if articles else {}
    st.write(f"  ✅ Sentiment={sentiment_data.get('sentiment_score')} in {time.time()-start:.1f}s")

    st.write("**STEP 3: UniverseScorer (options + squeeze)...**")
    start = time.time()
    from src.analytics.universe_scorer import UniverseScorer
    scorer = UniverseScorer(skip_ibkr=True)
    scores_list, _ = scorer.score_and_save_universe(tickers=[ticker], max_workers=1)
    st.write(f"  ✅ Done in {time.time()-start:.1f}s")

    st.write("**STEP 3c: Technical...**")
    start = time.time()
    ta = TechnicalAnalyzer()
    tech_result = ta.analyze_ticker(ticker)
    st.write(f"  ✅ Done in {time.time()-start:.1f}s")

    st.write("**STEP 6: Earnings (subprocess)...**")
    start = time.time()
    from src.analytics.yf_subprocess import get_earnings_dates
    ed = get_earnings_dates(ticker)
    st.write(f"  ✅ Done in {time.time()-start:.1f}s, has_data={ed is not None}")

    st.write("---")
    st.write("**✅ ALL STEPS COMPLETE - Scanner path works!**")
