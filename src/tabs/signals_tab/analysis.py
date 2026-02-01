"""
Signals Tab - Analysis Pipeline

Core analysis processing: _process_next, earnings-aware analysis, single ticker analysis.
"""

from .shared import (
    st, pd, np, logger, datetime, date, timedelta, time, json,
    Dict, List, Optional, _to_native, text,
    get_engine, get_connection,
    SIGNAL_HUB_AVAILABLE, DB_AVAILABLE,
    UNIVERSE_SCORER_AVAILABLE, UniverseScorer,
    NEWS_COLLECTOR_AVAILABLE, NewsCollector,
    SENTIMENT_ANALYZER_AVAILABLE, SentimentAnalyzer,
    TECHNICAL_ANALYZER_AVAILABLE, TechnicalAnalyzer,
    FUNDAMENTAL_ANALYZER_AVAILABLE, FundamentalAnalyzer,
    OPTIONS_FLOW_AVAILABLE, OptionsFlowAnalyzer,
    ENHANCED_SCORING_AVAILABLE, ENHANCED_SCORES_DB_AVAILABLE,
)

from .job_manager import _get_job_status, _complete_job, _log_result, _get_processed_tickers

if SIGNAL_HUB_AVAILABLE:
    from .shared import get_signal_engine, SignalEngine

if ENHANCED_SCORING_AVAILABLE:
    from .shared import get_enhanced_total_score

if ENHANCED_SCORES_DB_AVAILABLE:
    from src.analytics.enhanced_scores_db import compute_and_save_enhanced_scores

def _process_next(tickers: list, skip_if_analyzed_today: bool = True, force_fresh_news: bool = False):
    """
    Process next ticker with FULL analysis - runs everything:
    1. Fresh news collection
    2. Sentiment analysis
    3. Options flow, fundamentals, technical scores
    4. Create/update TODAY's row in screener_scores
    5. Update sentiment_scores table
    6. Regenerate signal with committee

    Args:
        tickers: List of tickers to process
        skip_if_analyzed_today: If True, skip tickers that already have today's row in screener_scores
        force_fresh_news: If True, force fetch new articles even if cache is fresh
    """
    from datetime import date as dt_date

    today = dt_date.today()

    # Get processed tickers from job log
    processed = _get_processed_tickers()

    # Calculate actual remaining (tickers not in job log)
    remaining_tickers = [t for t in tickers if t not in processed]

    # If no remaining tickers, we're done
    if not remaining_tickers:
        _complete_job()
        st.success("✅ All done!")
        st.balloons()
        time.sleep(2)
        st.rerun()
        return

    # =========================================================================
    # OPTIMIZATION: Batch check which tickers already have today's data
    # =========================================================================
    already_analyzed_today = set()
    analyzed_scores = {}  # ticker -> (sent, opts, fund, tech)

    if skip_if_analyzed_today:
        try:
            engine = get_engine()
            # Get ALL tickers with today's data in ONE query
            df_existing = pd.read_sql(f"""
                SELECT ticker, sentiment_score, options_flow_score, fundamental_score, technical_score
                FROM screener_scores 
                WHERE date = '{today}' AND sentiment_score IS NOT NULL
            """, engine)

            for _, row in df_existing.iterrows():
                t = row['ticker'].upper()
                already_analyzed_today.add(t)
                analyzed_scores[t] = (
                    row['sentiment_score'],
                    row['options_flow_score'],
                    row['fundamental_score'],
                    row['technical_score']
                )
            logger.info(f"Batch check: {len(already_analyzed_today)} tickers already have today's data")
        except Exception as e:
            logger.warning(f"Batch check failed: {e}")

    # =========================================================================
    # Find next ticker that ACTUALLY needs processing
    # Skip tickers already analyzed today (if option enabled)
    # =========================================================================
    # =========================================================================
    # Find next ticker that ACTUALLY needs processing
    # Skip tickers already analyzed today (if option enabled)
    # =========================================================================
    next_ticker = None
    skipped_count = 0

    for t in remaining_tickers:
        if t in already_analyzed_today:
            # Skip - already has today's data in screener_scores
            # Log to job_log with skip status so counts are accurate
            sent, opts, fund, tech = analyzed_scores.get(t, (None, None, None, None))
            logger.info(
                f"{t}: SKIPPED - already has today's data (Sent:{sent} Opts:{opts} Fund:{fund} Tech:{tech})")

            # Log skip to job_log using upsert
            try:
                engine = get_engine()
                with engine.connect() as conn:
                    result = conn.execute(text("""
                        INSERT INTO analysis_job_log (ticker, status, news_count, sentiment_score, options_score)
                        VALUES (:t, '⏭️', 0, :sent, :opts)
                        ON CONFLICT (ticker) DO UPDATE SET status = '⏭️'
                        RETURNING (xmax = 0) AS inserted
                    """), {'t': t, 'sent': _to_native(sent), 'opts': _to_native(opts)})
                    row = result.fetchone()
                    if row and row[0]:  # was inserted
                        conn.execute(
                            text("UPDATE analysis_job SET processed_count=processed_count+1, updated_at=NOW()"))
                    conn.commit()
            except Exception as e:
                logger.debug(f"{t}: Skip log failed: {e}")

            skipped_count += 1
            continue
        # Found a ticker that needs processing
        next_ticker = t
        break
    # If we skipped some tickers, log it
    if skipped_count > 0:
        logger.info(f"Batch skipped {skipped_count} tickers that were already analyzed today")

    if not next_ticker:
        _complete_job()
        st.success("✅ All done!")
        st.balloons()
        time.sleep(2)
        st.rerun()
        return

    # Progress display
    total_processed = len(processed)
    # Note: Progress bar is rendered in _render_analysis_panel(), don't duplicate here
    st.info(f"🔄 Processing **{next_ticker}** ({total_processed + 1}/{len(tickers)})")
    # CRITICAL: Pre-log ticker as "in progress" BEFORE processing
    # This prevents infinite loops if processing fails
    # =========================================================================
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Use upsert to avoid race conditions - only increment count on actual insert
            result = conn.execute(text("""
                INSERT INTO analysis_job_log (ticker, status, news_count)
                VALUES (:t, '🔄', 0)
                ON CONFLICT (ticker) DO UPDATE SET status = '🔄'
                RETURNING (xmax = 0) AS inserted
            """), {'t': next_ticker})
            row = result.fetchone()
            was_inserted = row[0] if row else False

            # Only increment processed_count if this was a new insert
            if was_inserted:
                conn.execute(text("UPDATE analysis_job SET processed_count=processed_count+1, updated_at=NOW()"))
                logger.info(f"{next_ticker}: Pre-logged as in-progress")
            else:
                logger.debug(f"{next_ticker}: Already in job log, updating status")
            conn.commit()
    except Exception as e:
        logger.warning(f"{next_ticker}: Pre-log failed: {e}")
        # If pre-log fails, we should still continue but log the error

    status = '✅'
    news_count = 0
    sentiment = None
    options = None
    fundamental = None
    technical = None
    committee_verdict = None
    today = dt_date.today()

    try:
        # =====================================================================
        # STEP 1: Collect Fresh News
        # =====================================================================
        nc = NewsCollector()
        result = nc.collect_and_save(next_ticker, days_back=7, force_refresh=force_fresh_news)
        articles = result.get('articles', [])
        news_count = len(articles)

        # =====================================================================
        # STEP 2: Analyze Sentiment
        # =====================================================================
        sentiment_data = {}
        if articles:
            sa = SentimentAnalyzer()
            sentiment_data = sa.analyze_ticker_sentiment(next_ticker, articles)
            sentiment = sentiment_data.get('sentiment_score')  # None if not present
        else:
            sentiment = None  # FIX: Don't use 50 as default - let it be None to indicate missing data

        # =====================================================================
        # STEP 3: Get Options Flow & Squeeze Scores (from UniverseScorer)
        # =====================================================================
        squeeze = None
        growth = None
        dividend = None

        try:
            scorer = UniverseScorer()
            scores_list, _ = scorer.score_and_save_universe(tickers=[next_ticker], max_workers=1)

            # FIX: scores_list is a List[UniverseScores], not a dict!
            # Convert to dict for easier access
            if scores_list:
                for score_obj in scores_list:
                    if score_obj.ticker == next_ticker:
                        # UniverseScores only has: options_flow_score, short_squeeze_score
                        # It does NOT have fundamental/technical/growth/dividend
                        options = score_obj.options_flow_score
                        squeeze = score_obj.short_squeeze_score
                        break
        except Exception as e:
            logger.warning(f"{next_ticker}: Universe scorer error - {e}")

        # =====================================================================
        # STEP 3b: Get Fundamental Score (from database)
        # =====================================================================
        try:
            # FundamentalAnalyzer not implemented - scores come from screener
            pass
        except Exception as e:
            logger.debug(f"{next_ticker}: Fundamental analysis skipped - {e}")

        # =====================================================================
        # STEP 3c: Get Technical Score (from TechnicalAnalyzer)
        # =====================================================================
        try:
            ta = TechnicalAnalyzer()
            tech_result = ta.analyze_ticker(next_ticker)
            if tech_result:
                technical = tech_result.get('technical_score', tech_result.get('score'))
        except Exception as e:
            logger.debug(f"{next_ticker}: Technical analysis skipped - {e}")

        # =====================================================================
        # STEP 4: Update sentiment_scores table
        # =====================================================================
        # Determine sentiment class
        # BUG FIX: Guard against None to prevent TypeError crash
        if sentiment is None:
            sentiment_class = 'Unknown'
        elif sentiment >= 70:
            sentiment_class = 'Very Bullish'
        elif sentiment >= 55:
            sentiment_class = 'Bullish'
        elif sentiment >= 45:
            sentiment_class = 'Neutral'
        elif sentiment >= 30:
            sentiment_class = 'Bearish'
        else:
            sentiment_class = 'Very Bearish'

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO sentiment_scores
                        (ticker, date, sentiment_raw, sentiment_weighted, ai_sentiment_fast,
                         article_count, relevant_article_count, sentiment_class)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) 
                        ON CONFLICT (ticker, date) DO UPDATE SET
                            sentiment_raw = EXCLUDED.sentiment_raw,
                            sentiment_weighted = EXCLUDED.sentiment_weighted,
                            ai_sentiment_fast = EXCLUDED.ai_sentiment_fast,
                            article_count = EXCLUDED.article_count,
                            relevant_article_count = EXCLUDED.relevant_article_count,
                            sentiment_class = EXCLUDED.sentiment_class
                    """, (
                        next_ticker,
                        today,
                        _to_native(sentiment),
                        _to_native(sentiment_data.get('sentiment_weighted', sentiment)),
                        _to_native(sentiment),
                        _to_native(news_count),
                        _to_native(sentiment_data.get('relevant_count', news_count)),
                        sentiment_class
                    ))
                conn.commit()
        except Exception as e:
            logger.warning(f"{next_ticker}: sentiment_scores update error - {e}")

        # =====================================================================
        # STEP 5: Create/Update TODAY's row in screener_scores
        # =====================================================================
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if today's row exists
                    cur.execute("""
                        SELECT COUNT(*) FROM screener_scores 
                        WHERE ticker = %s AND date = %s
                    """, (next_ticker, today))
                    today_exists = cur.fetchone()[0] > 0

                    if today_exists:
                        # Update existing row
                        cur.execute("""
                            UPDATE screener_scores SET
                                sentiment_score = %s,
                                article_count = %s,
                                sentiment_weighted = %s,
                                options_flow_score = COALESCE(%s, options_flow_score),
                                short_squeeze_score = COALESCE(%s, short_squeeze_score),
                                fundamental_score = COALESCE(%s, fundamental_score),
                                technical_score = COALESCE(%s, technical_score),
                                growth_score = COALESCE(%s, growth_score),
                                dividend_score = COALESCE(%s, dividend_score)
                            WHERE ticker = %s AND date = %s
                        """, (
                            _to_native(sentiment), _to_native(news_count), _to_native(sentiment_data.get('sentiment_weighted', sentiment)),
                            _to_native(options), _to_native(squeeze), _to_native(fundamental), _to_native(technical), _to_native(growth), _to_native(dividend),
                            next_ticker, today
                        ))
                    else:
                        # Create new row for today by copying from most recent
                        cur.execute("""
                            SELECT * FROM screener_scores 
                            WHERE ticker = %s 
                            ORDER BY date DESC LIMIT 1
                        """, (next_ticker,))

                        existing_row = cur.fetchone()

                        if existing_row:
                            col_names = [desc[0] for desc in cur.description]
                            existing_data = dict(zip(col_names, existing_row))

                            # FIX Issue #4: Compute total_score fresh instead of copying stale value
                            # Use today's scores where available, fall back to existing only if needed
                            # FIX: Use None instead of 50 to indicate missing data
                            final_sentiment = sentiment if sentiment is not None else existing_data.get('sentiment_score')
                            final_fundamental = fundamental if fundamental is not None else existing_data.get('fundamental_score')
                            final_technical = technical if technical is not None else existing_data.get('technical_score')
                            final_options = options if options is not None else existing_data.get('options_flow_score')
                            final_squeeze = squeeze if squeeze is not None else existing_data.get('short_squeeze_score')
                            final_growth = growth if growth is not None else existing_data.get('growth_score')
                            final_dividend = dividend if dividend is not None else existing_data.get('dividend_score')

                            # Recompute total_score using weighted average of available scores
                            # Weights: sentiment=25%, fundamental=25%, technical=25%, options=15%, squeeze=10%
                            # FIX: Handle None values - only include scores that are available
                            available_scores = []
                            available_weights = []

                            if final_sentiment is not None:
                                available_scores.append(final_sentiment * 0.25)
                                available_weights.append(0.25)
                            if final_fundamental is not None:
                                available_scores.append(final_fundamental * 0.25)
                                available_weights.append(0.25)
                            if final_technical is not None:
                                available_scores.append(final_technical * 0.25)
                                available_weights.append(0.25)
                            if final_options is not None:
                                available_scores.append(final_options * 0.15)
                                available_weights.append(0.15)
                            if final_squeeze is not None:
                                available_scores.append(final_squeeze * 0.10)
                                available_weights.append(0.10)

                            # Calculate base_score as weighted average of available scores
                            if available_weights:
                                total_weight = sum(available_weights)
                                # BUG FIX: Use round() instead of int() to avoid systematic downward bias
                                base_score = round(sum(available_scores) / total_weight) if total_weight > 0 else None
                                if base_score is not None:
                                    base_score = max(0, min(100, base_score))  # Clamp to valid range
                            else:
                                base_score = None

                            # Apply enhanced scoring adjustments if available
                            total_score = base_score
                            if ENHANCED_SCORING_AVAILABLE:
                                try:
                                    # Build row_data for enhanced scoring
                                    enhanced_row_data = {
                                        'price': existing_data.get('price'),
                                        'target_mean': existing_data.get('target_mean'),
                                        'pe_ratio': existing_data.get('pe_ratio'),
                                        'forward_pe': existing_data.get('forward_pe'),
                                        'peg_ratio': existing_data.get('peg_ratio'),
                                        'sector': existing_data.get('sector'),
                                        'buy_count': existing_data.get('buy_count', 0),
                                        'total_ratings': existing_data.get('total_ratings', 0),
                                    }
                                    enhanced_score, adjustment, _ = get_enhanced_total_score(
                                        ticker=next_ticker,
                                        base_score=base_score,
                                        row_data=enhanced_row_data,
                                    )
                                    total_score = enhanced_score
                                    if adjustment != 0:
                                        logger.info(f"{next_ticker}: Enhanced score {base_score} → {enhanced_score} (adj: {adjustment:+d})")
                                except Exception as e:
                                    logger.debug(f"{next_ticker}: Enhanced scoring skipped: {e}")

                            # FIX Issue #5: Track data quality - count how many scores are fresh vs stale
                            fresh_count = sum([
                                1 if sentiment is not None else 0,
                                1 if fundamental is not None else 0,
                                1 if technical is not None else 0,
                                1 if options is not None else 0,
                            ])
                            if fresh_count < 2:
                                logger.warning(f"{next_ticker}: Low data quality - only {fresh_count}/4 scores are fresh")

                            cur.execute("""
                                INSERT INTO screener_scores (
                                    ticker, date, 
                                    sentiment_score, sentiment_weighted, article_count,
                                    fundamental_score, technical_score, growth_score, dividend_score,
                                    gap_score, likelihood_score, total_score,
                                    options_flow_score, short_squeeze_score, options_sentiment, squeeze_risk,
                                    created_at
                                ) VALUES (
                                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                                )
                            """, (
                                next_ticker, today,
                                _to_native(final_sentiment),
                                _to_native(sentiment_data.get('sentiment_weighted', final_sentiment)),
                                _to_native(news_count),
                                _to_native(final_fundamental),
                                _to_native(final_technical),
                                _to_native(final_growth),
                                _to_native(final_dividend),
                                _to_native(existing_data.get('gap_score', 50)),
                                _to_native(existing_data.get('likelihood_score', 50)),
                                _to_native(total_score),
                                _to_native(final_options),
                                _to_native(final_squeeze),
                                _to_native(existing_data.get('options_sentiment')),
                                _to_native(existing_data.get('squeeze_risk')),
                            ))

                            # Save enhanced scores to separate table for fast deep-dive loading
                            if ENHANCED_SCORES_DB_AVAILABLE:
                                try:
                                    compute_and_save_enhanced_scores(
                                        ticker=next_ticker,
                                        row_data=enhanced_row_data if 'enhanced_row_data' in dir() else existing_data,
                                    )
                                except Exception as e:
                                    logger.debug(f"{next_ticker}: Could not save enhanced scores: {e}")
                        else:
                            # No existing row - create new one with computed total_score
                            # FIX: Keep None values to indicate missing data, don't default to 50
                            final_sentiment = sentiment  # Keep as is (None if missing)
                            final_fundamental = fundamental
                            final_technical = technical
                            final_options = options
                            final_squeeze = squeeze
                            final_growth = growth
                            final_dividend = dividend

                            # Compute base total_score - handle None values
                            available_scores = []
                            available_weights = []

                            if final_sentiment is not None:
                                available_scores.append(final_sentiment * 0.25)
                                available_weights.append(0.25)
                            if final_fundamental is not None:
                                available_scores.append(final_fundamental * 0.25)
                                available_weights.append(0.25)
                            if final_technical is not None:
                                available_scores.append(final_technical * 0.25)
                                available_weights.append(0.25)
                            if final_options is not None:
                                available_scores.append(final_options * 0.15)
                                available_weights.append(0.15)
                            if final_squeeze is not None:
                                available_scores.append(final_squeeze * 0.10)
                                available_weights.append(0.10)

                            if available_weights:
                                total_weight = sum(available_weights)
                                # BUG FIX: Use round() instead of int() to avoid systematic downward bias
                                base_score = round(sum(available_scores) / total_weight) if total_weight > 0 else None
                                if base_score is not None:
                                    base_score = max(0, min(100, base_score))  # Clamp to valid range
                            else:
                                base_score = None

                            # Apply enhanced scoring if available (limited data for new ticker)
                            total_score = base_score
                            if ENHANCED_SCORING_AVAILABLE:
                                try:
                                    enhanced_score, adjustment, _ = get_enhanced_total_score(
                                        ticker=next_ticker,
                                        base_score=base_score,
                                        row_data={},  # No existing data
                                    )
                                    total_score = enhanced_score
                                except:
                                    pass  # Use base score if enhanced fails

                            cur.execute("""
                                INSERT INTO screener_scores (
                                    ticker, date, 
                                    sentiment_score, sentiment_weighted, article_count,
                                    fundamental_score, technical_score, growth_score, dividend_score,
                                    total_score, options_flow_score, short_squeeze_score,
                                    created_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                            """, (
                                next_ticker, today,
                                _to_native(final_sentiment),
                                _to_native(sentiment_data.get('sentiment_weighted', final_sentiment)),
                                _to_native(news_count),
                                _to_native(final_fundamental),
                                _to_native(final_technical),
                                _to_native(final_growth),
                                _to_native(final_dividend),
                                _to_native(total_score),
                                _to_native(final_options),
                                _to_native(final_squeeze),
                            ))

                    conn.commit()
                    logger.info(f"{next_ticker}: screener_scores updated for {today}")

                    # Auto-sync to historical_scores for backtesting
                    try:
                        from src.utils.historical_sync import sync_to_historical
                        sync_to_historical(next_ticker, today, {
                            'sentiment_score': final_sentiment if 'final_sentiment' in dir() else sentiment,
                            'fundamental_score': final_fundamental if 'final_fundamental' in dir() else fundamental,
                            'growth_score': final_growth if 'final_growth' in dir() else growth,
                            'dividend_score': final_dividend if 'final_dividend' in dir() else dividend,
                            'total_score': total_score,
                            'gap_score': existing_data.get('gap_score') if 'existing_data' in dir() and existing_data else None,
                            'composite_score': existing_data.get('composite_score') if 'existing_data' in dir() and existing_data else None,
                        })
                    except Exception as e:
                        logger.debug(f"{next_ticker}: historical sync skipped: {e}")
        except Exception as e:
            logger.warning(f"{next_ticker}: screener_scores update error - {e}")

        # =====================================================================
        # STEP 6: Update Earnings Calendar from yfinance
        # =====================================================================
        try:
            import yfinance as yf
            stock = yf.Ticker(next_ticker)
            ed = stock.earnings_dates

            if ed is not None and not ed.empty:
                # Get the next upcoming earnings date
                for idx in ed.index:
                    earnings_dt = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                    if earnings_dt >= today:
                        # Found next earnings date - save to DB
                        with get_connection() as conn:
                            with conn.cursor() as cur:
                                # Try with updated_at, fall back to simpler insert
                                try:
                                    cur.execute("""
                                        INSERT INTO earnings_calendar (ticker, earnings_date, updated_at)
                                        VALUES (%s, %s, NOW())
                                        ON CONFLICT (ticker) DO UPDATE SET
                                            earnings_date = EXCLUDED.earnings_date,
                                            updated_at = NOW()
                                    """, (next_ticker, earnings_dt))
                                except Exception:
                                    # Table might not have updated_at column
                                    cur.execute("""
                                        INSERT INTO earnings_calendar (ticker, earnings_date)
                                        VALUES (%s, %s)
                                        ON CONFLICT (ticker) DO UPDATE SET
                                            earnings_date = EXCLUDED.earnings_date
                                    """, (next_ticker, earnings_dt))
                            conn.commit()
                        logger.debug(f"{next_ticker}: Updated earnings_calendar -> {earnings_dt}")
                        break
        except Exception as e:
            logger.debug(f"{next_ticker}: Could not update earnings calendar: {e}")

        # =====================================================================
        # STEP 7: Derive Signal Verdict from Total Score
        # NOTE: We skip calling generate_signal() here because it re-runs ALL
        # analysis (options, earnings, 13F) which we already computed above.
        # The verdict is simply derived from the total_score we just saved.
        # =====================================================================
        try:
            # Derive committee verdict from total_score
            if total_score is not None:
                if total_score >= 80:
                    committee_verdict = 'STRONG_BUY'
                elif total_score >= 65:
                    committee_verdict = 'BUY'
                elif total_score >= 55:
                    committee_verdict = 'WEAK_BUY'
                elif total_score <= 20:
                    committee_verdict = 'STRONG_SELL'
                elif total_score <= 35:
                    committee_verdict = 'SELL'
                elif total_score <= 45:
                    committee_verdict = 'WEAK_SELL'
                else:
                    committee_verdict = 'HOLD'
            else:
                committee_verdict = 'HOLD'
            
            # Save the signal to trading_signals table
            try:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO trading_signals (ticker, date, signal_type, score, created_at)
                            VALUES (%s, %s, %s, %s, NOW())
                            ON CONFLICT (ticker, date) DO UPDATE SET
                                signal_type = EXCLUDED.signal_type,
                                score = EXCLUDED.score,
                                created_at = NOW()
                        """, (next_ticker, today, committee_verdict, total_score))
                    conn.commit()
            except Exception as e:
                logger.debug(f"{next_ticker}: trading_signals update skipped: {e}")
            
            # Clear SignalEngine cache so next load gets fresh data
            try:
                from src.core import get_signal_engine
                engine = get_signal_engine()
                if hasattr(engine, '_cache'):
                    for t in [next_ticker, next_ticker.upper(), next_ticker.lower()]:
                        if t in engine._cache:
                            del engine._cache[t]
            except:
                pass
            
            logger.info(f"{next_ticker}: Signal regenerated - Committee: {committee_verdict}")
        except Exception as e:
            logger.warning(f"{next_ticker}: Signal derivation error - {e}")
            committee_verdict = 'HOLD'

    except Exception as e:
        status = '⚠️'
        logger.error(f"Error processing {next_ticker}: {e}")

    # Log result with more details
    _log_result(next_ticker, status, news_count, sentiment, options, fundamental, technical, committee_verdict)
    st.rerun()


def _check_earnings_status(ticker: str) -> str:
    """Check if ticker is near earnings. Returns 'pre', 'post', or 'none'."""
    try:
        import yfinance as yf

        engine = get_engine()

        # Check database for earnings date
        df = pd.read_sql(f"""
            SELECT earnings_date FROM earnings_calendar 
            WHERE ticker = '{ticker}' 
            AND earnings_date >= CURRENT_DATE - INTERVAL '5 days'
            AND earnings_date <= CURRENT_DATE + INTERVAL '5 days'
            ORDER BY ABS(earnings_date - CURRENT_DATE) LIMIT 1
        """, engine)

        if not df.empty and df.iloc[0]['earnings_date']:
            ed = pd.to_datetime(df.iloc[0]['earnings_date']).date()
            days = (ed - date.today()).days

            if days < 0:
                return 'post'
            elif days <= 5:
                return 'pre'

        # Fallback to yfinance
        stock = yf.Ticker(ticker)
        try:
            hist = stock.earnings_history
            if hist is not None and not hist.empty:
                latest_date = pd.to_datetime(hist.index[0]).date()
                days = (latest_date - date.today()).days
                if days >= -5 and days <= 0:
                    return 'post'
        except:
            pass

    except Exception as e:
        logger.debug(f"Earnings status check error for {ticker}: {e}")

    return 'none'


def _run_earnings_aware_analysis(ticker: str, earnings_status: str) -> Dict:
    """Run analysis with earnings context."""
    import yfinance as yf

    result = {
        'ticker': ticker,
        'news_count': 0,
        'sentiment_score': None,
        'earnings_summary': None,
    }

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get('shortName', info.get('longName', ticker))

        # Build earnings-specific search queries
        if earnings_status == 'post':
            queries = [
                f"{ticker} earnings results",
                f"{company_name} quarterly earnings",
                f"{ticker} earnings reaction",
                f"{ticker} guidance outlook",
            ]
        else:
            queries = [
                f"{ticker} earnings preview expectations",
                f"{company_name} earnings whisper",
            ]

        nc = NewsCollector()
        all_articles = []

        # Collect with earnings queries
        for query in queries:
            try:
                articles = nc.collect_ai_search(ticker, company_name=query)
                for a in articles:
                    a['ticker'] = ticker
                all_articles.extend(articles)
            except:
                pass

        # Also standard collection with force refresh
        standard = nc.collect_and_save(ticker, days_back=5, force_refresh=True)
        all_articles.extend(standard.get('articles', []))

        # Deduplicate
        seen = set()
        unique = []
        for a in all_articles:
            title = str(a.get('title', '')).lower()[:40]
            if title and title not in seen:
                seen.add(title)
                unique.append(a)

        result['news_count'] = len(unique)

        # Save and analyze sentiment
        if unique:
            nc.save_articles(unique)

            sa = SentimentAnalyzer()
            sent_result = sa.analyze_ticker_sentiment(ticker, unique)
            if sent_result:
                result['sentiment_score'] = sent_result.get('sentiment_score')

        # Get earnings result if post-earnings
        if earnings_status == 'post':
            try:
                hist = stock.earnings_history
                if hist is not None and not hist.empty:
                    latest = hist.iloc[0]
                    eps_actual = latest.get('epsActual')
                    eps_est = latest.get('epsEstimate')

                    surprise_pct = None
                    if eps_actual and eps_est and eps_est != 0:
                        surprise_pct = ((eps_actual - eps_est) / abs(eps_est)) * 100

                    # Get price reaction
                    price_hist = stock.history(period="5d")
                    reaction_pct = 0
                    if len(price_hist) >= 2:
                        reaction_pct = ((price_hist['Close'].iloc[-1] - price_hist['Close'].iloc[-2]) /
                                        price_hist['Close'].iloc[-2]) * 100

                    overall = "MISS" if (eps_actual or 0) < (eps_est or 0) else "BEAT" if (eps_actual or 0) > (
                                eps_est or 0) else "INLINE"

                    result['earnings_summary'] = {
                        'eps_actual': eps_actual,
                        'eps_estimate': eps_est,
                        'eps_surprise_pct': surprise_pct,
                        'reaction_pct': reaction_pct,
                        'overall_result': overall,
                    }
            except Exception as e:
                logger.debug(f"Earnings result error: {e}")

    except Exception as e:
        logger.error(f"Earnings-aware analysis error for {ticker}: {e}")

    return result


# =============================================================================
# SIGNALS VIEW
# =============================================================================



# =============================================================================
# SINGLE TICKER ANALYSIS
# =============================================================================

def _run_single_analysis(ticker: str):
    """
    Run FULL analysis for single ticker - refresh everything automatically.

    This function:
    1. Fetches fresh news
    2. Analyzes sentiment
    3. Gets options flow, fundamentals, technical scores
    4. Creates/updates a row for TODAY in screener_scores
    5. Clears all caches
    6. Forces page refresh to show new data
    """
    with st.spinner(f"🔄 Refreshing all data for {ticker}..."):
        try:
            from datetime import date as dt_date

            status_container = st.empty()
            results_container = st.container()
            today = dt_date.today()

            # =====================================================================
            # STEP 1: Collect Fresh News
            # =====================================================================
            status_container.info(f"📰 Step 1/6: Collecting fresh news for {ticker}...")
            nc = NewsCollector()
            result = nc.collect_and_save(ticker, days_back=7, force_refresh=True)
            articles = result.get('articles', [])
            saved_count = result.get('saved', 0)

            # =====================================================================
            # STEP 2: Analyze Sentiment
            # =====================================================================
            status_container.info(f"🧠 Step 2/6: Analyzing sentiment ({len(articles)} articles)...")
            sentiment_score = None  # FIX: Use None, not 50
            sentiment_data = {}
            if articles:
                sa = SentimentAnalyzer()
                sentiment_data = sa.analyze_ticker_sentiment(ticker, articles)
                sentiment_score = sentiment_data.get('sentiment_score')  # FIX: No default

            # =====================================================================
            # STEP 3: Get Options Flow & Other Scores
            # =====================================================================
            status_container.info(f"📊 Step 3/6: Updating options flow & fundamentals...")
            options_score = None
            squeeze_score = None
            fundamental_score = None
            technical_score = None
            growth_score = None
            dividend_score = None

            try:
                scorer = UniverseScorer()
                scores_list, _ = scorer.score_and_save_universe(tickers=[ticker], max_workers=1)

                # FIX: scores_list is a List[UniverseScores], not a dict!
                if scores_list:
                    for score_obj in scores_list:
                        if score_obj.ticker == ticker:
                            # UniverseScores only has options/squeeze, not fundamental/technical
                            options_score = score_obj.options_flow_score
                            squeeze_score = score_obj.short_squeeze_score
                            break
            except Exception as e:
                logger.warning(f"Universe scorer error for {ticker}: {e}")

            # Get Fundamental Score separately
            try:
                # FundamentalAnalyzer not implemented - scores come from screener
                pass
            except Exception as e:
                logger.debug(f"{ticker}: Fundamental analysis skipped - {e}")

            # Get Technical Score separately
            try:
                ta = TechnicalAnalyzer()
                tech_result = ta.analyze_ticker(ticker)
                if tech_result:
                    technical_score = tech_result.get('technical_score', tech_result.get('score'))
            except Exception as e:
                logger.debug(f"{ticker}: Technical analysis skipped - {e}")

            # =====================================================================
            # STEP 4: Update sentiment_scores table (UPSERT)
            # =====================================================================
            status_container.info(f"💾 Step 4/6: Saving sentiment scores...")

            # Determine sentiment class
            if sentiment_score >= 70:
                sentiment_class = 'Very Bullish'
            elif sentiment_score >= 55:
                sentiment_class = 'Bullish'
            elif sentiment_score >= 45:
                sentiment_class = 'Neutral'
            elif sentiment_score >= 30:
                sentiment_class = 'Bearish'
            else:
                sentiment_class = 'Very Bearish'

            with get_connection() as conn:
                with conn.cursor() as cur:
                    # UPSERT sentiment_scores
                    cur.execute("""
                        INSERT INTO sentiment_scores
                        (ticker, date, sentiment_raw, sentiment_weighted, ai_sentiment_fast,
                         article_count, relevant_article_count, sentiment_class)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) 
                        ON CONFLICT (ticker, date) DO UPDATE SET
                            sentiment_raw = EXCLUDED.sentiment_raw,
                            sentiment_weighted = EXCLUDED.sentiment_weighted,
                            ai_sentiment_fast = EXCLUDED.ai_sentiment_fast,
                            article_count = EXCLUDED.article_count,
                            relevant_article_count = EXCLUDED.relevant_article_count,
                            sentiment_class = EXCLUDED.sentiment_class
                    """, (
                        ticker,
                        today,
                        sentiment_score,
                        sentiment_data.get('sentiment_weighted', sentiment_score),
                        sentiment_score,
                        len(articles),
                        sentiment_data.get('relevant_count', len(articles)),
                        sentiment_class
                    ))
                conn.commit()

            # =====================================================================
            # STEP 5: Create/Update TODAY's row in screener_scores
            # =====================================================================
            status_container.info(f"💾 Step 5/6: Updating screener_scores for today...")

            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if today's row exists
                    cur.execute("""
                        SELECT COUNT(*) FROM screener_scores 
                        WHERE ticker = %s AND date = %s
                    """, (ticker, today))
                    today_exists = cur.fetchone()[0] > 0

                    if today_exists:
                        # Update existing row for today
                        logger.info(f"{ticker}: Updating existing screener_scores row for {today}")

                        # First get current values to compute total_score
                        cur.execute("""
                            SELECT sentiment_score, fundamental_score, technical_score, 
                                   options_flow_score, short_squeeze_score
                            FROM screener_scores WHERE ticker = %s AND date = %s
                        """, (ticker, today))
                        current = cur.fetchone()

                        # Compute fresh total_score using new values where available
                        # BUG FIX: Use "is not None" - score of 0 is valid!
                        final_sent = sentiment_score if sentiment_score is not None else (current[0] if current else 50)
                        final_fund = fundamental_score if fundamental_score is not None else (current[1] if current else 50)
                        final_tech = technical_score if technical_score is not None else (current[2] if current else 50)
                        final_opts = options_score if options_score is not None else (current[3] if current else 50)
                        final_sqz = squeeze_score if squeeze_score is not None else (current[4] if current else 50)

                        # Weighted average: sent=25%, fund=25%, tech=25%, opts=15%, sqz=10%
                        # BUG FIX: Use round() instead of int() to avoid systematic downward bias
                        base_total = round(
                            final_sent * 0.25 + final_fund * 0.25 + final_tech * 0.25 +
                            final_opts * 0.15 + final_sqz * 0.10
                        )
                        # Clamp to valid range
                        base_total = max(0, min(100, base_total))

                        # Apply enhanced scoring if available
                        computed_total = base_total
                        if ENHANCED_SCORING_AVAILABLE:
                            try:
                                enhanced_score, adjustment, _ = get_enhanced_total_score(
                                    ticker=ticker,
                                    base_score=base_total,
                                    row_data={},
                                )
                                computed_total = enhanced_score
                                if adjustment != 0:
                                    logger.info(f"{ticker}: Enhanced score {base_total} → {computed_total} (adj: {adjustment:+d})")
                            except:
                                pass

                        cur.execute("""
                            UPDATE screener_scores SET
                                sentiment_score = %s,
                                article_count = %s,
                                sentiment_weighted = %s,
                                options_flow_score = COALESCE(%s, options_flow_score),
                                short_squeeze_score = COALESCE(%s, short_squeeze_score),
                                fundamental_score = COALESCE(%s, fundamental_score),
                                technical_score = COALESCE(%s, technical_score),
                                growth_score = COALESCE(%s, growth_score),
                                dividend_score = COALESCE(%s, dividend_score),
                                total_score = %s
                            WHERE ticker = %s AND date = %s
                        """, (
                            _to_native(sentiment_score),
                            _to_native(len(articles)),
                            _to_native(sentiment_data.get('sentiment_weighted', sentiment_score)),
                            _to_native(options_score),
                            _to_native(squeeze_score),
                            _to_native(fundamental_score),
                            _to_native(technical_score),
                            _to_native(growth_score),
                            _to_native(dividend_score),
                            _to_native(computed_total),
                            ticker,
                            today
                        ))
                    else:
                        # Create new row for today by copying from most recent row
                        logger.info(f"{ticker}: Creating new screener_scores row for {today} (copying from most recent)")

                        # First, get the most recent row to copy base values
                        cur.execute("""
                            SELECT * FROM screener_scores 
                            WHERE ticker = %s 
                            ORDER BY date DESC LIMIT 1
                        """, (ticker,))

                        existing_row = cur.fetchone()

                        if existing_row:
                            # Get column names
                            col_names = [desc[0] for desc in cur.description]
                            existing_data = dict(zip(col_names, existing_row))

                            # FIX Issue #4: Compute total_score fresh
                            # FIX: Use None instead of 50 for missing data
                            final_sent = sentiment_score if sentiment_score is not None else existing_data.get('sentiment_score')
                            final_fund = fundamental_score if fundamental_score is not None else existing_data.get('fundamental_score')
                            final_tech = technical_score if technical_score is not None else existing_data.get('technical_score')
                            final_opts = options_score if options_score is not None else existing_data.get('options_flow_score')
                            final_sqz = squeeze_score if squeeze_score is not None else existing_data.get('short_squeeze_score')
                            final_growth = growth_score if growth_score is not None else existing_data.get('growth_score')
                            final_div = dividend_score if dividend_score is not None else existing_data.get('dividend_score')

                            # Weighted average: only include available scores
                            available_scores = []
                            available_weights = []
                            if final_sent is not None:
                                available_scores.append(final_sent * 0.25)
                                available_weights.append(0.25)
                            if final_fund is not None:
                                available_scores.append(final_fund * 0.25)
                                available_weights.append(0.25)
                            if final_tech is not None:
                                available_scores.append(final_tech * 0.25)
                                available_weights.append(0.25)
                            if final_opts is not None:
                                available_scores.append(final_opts * 0.15)
                                available_weights.append(0.15)
                            if final_sqz is not None:
                                available_scores.append(final_sqz * 0.10)
                                available_weights.append(0.10)

                            if available_weights:
                                total_weight = sum(available_weights)
                                # BUG FIX: Use round() instead of int() to avoid systematic downward bias
                                base_total = round(sum(available_scores) / total_weight) if total_weight > 0 else None
                                if base_total is not None:
                                    base_total = max(0, min(100, base_total))  # Clamp to valid range
                            else:
                                base_total = None

                            # Apply enhanced scoring if available
                            computed_total = base_total
                            if ENHANCED_SCORING_AVAILABLE:
                                try:
                                    enhanced_row_data = {
                                        'price': existing_data.get('price'),
                                        'target_mean': existing_data.get('target_mean'),
                                        'pe_ratio': existing_data.get('pe_ratio'),
                                        'forward_pe': existing_data.get('forward_pe'),
                                        'peg_ratio': existing_data.get('peg_ratio'),
                                        'sector': existing_data.get('sector'),
                                        'buy_count': existing_data.get('buy_count', 0),
                                        'total_ratings': existing_data.get('total_ratings', 0),
                                    }
                                    enhanced_score, adjustment, _ = get_enhanced_total_score(
                                        ticker=ticker,
                                        base_score=base_total,
                                        row_data=enhanced_row_data,
                                    )
                                    computed_total = enhanced_score
                                    if adjustment != 0:
                                        logger.info(f"{ticker}: Enhanced score {base_total} → {computed_total} (adj: {adjustment:+d})")
                                except:
                                    pass

                            # FIX Issue #5: Track data freshness
                            # BUG FIX: Use "is not None" instead of truthiness
                            # Because score of 0 is valid but would be treated as "not fresh"
                            fresh_count = sum([
                                1 if sentiment_score is not None else 0,
                                1 if fundamental_score is not None else 0,
                                1 if technical_score is not None else 0,
                                1 if options_score is not None else 0,
                            ])
                            if fresh_count < 2:
                                logger.warning(f"{ticker}: Low data quality - only {fresh_count}/4 scores are fresh")

                            # Insert new row with COMPUTED total_score
                            cur.execute("""
                                INSERT INTO screener_scores (
                                    ticker, date, 
                                    sentiment_score, sentiment_weighted, article_count,
                                    fundamental_score, technical_score, growth_score, dividend_score,
                                    gap_score, likelihood_score, total_score,
                                    options_flow_score, short_squeeze_score, options_sentiment, squeeze_risk,
                                    created_at
                                ) VALUES (
                                    %s, %s,
                                    %s, %s, %s,
                                    %s, %s, %s, %s,
                                    %s, %s, %s,
                                    %s, %s, %s, %s,
                                    NOW()
                                )
                            """, (
                                ticker, today,
                                _to_native(final_sent),
                                _to_native(sentiment_data.get('sentiment_weighted', final_sent)),
                                _to_native(len(articles)),
                                _to_native(final_fund),
                                _to_native(final_tech),
                                _to_native(final_growth),
                                _to_native(final_div),
                                _to_native(existing_data.get('gap_score', 50)),
                                _to_native(existing_data.get('likelihood_score', 50)),
                                _to_native(computed_total),
                                _to_native(final_opts),
                                _to_native(final_sqz),
                                _to_native(existing_data.get('options_sentiment')),
                                _to_native(existing_data.get('squeeze_risk')),
                            ))
                            logger.info(f"{ticker}: Created new row for {today} with fresh total_score={computed_total}")

                            # Save enhanced scores for fast deep-dive loading
                            if ENHANCED_SCORES_DB_AVAILABLE:
                                try:
                                    compute_and_save_enhanced_scores(ticker=ticker, row_data=existing_data)
                                except Exception as e:
                                    logger.debug(f"{ticker}: Could not save enhanced scores: {e}")
                        else:
                            # No existing row at all - create a new one with computed total_score
                            logger.warning(f"{ticker}: No existing screener_scores rows, creating new row")

                            # Use available values or default to 50
                            # BUG FIX: Use "is not None" - score of 0 is valid!
                            final_sent = sentiment_score if sentiment_score is not None else 50
                            final_fund = fundamental_score if fundamental_score is not None else 50
                            final_tech = technical_score if technical_score is not None else 50
                            final_opts = options_score if options_score is not None else 50
                            final_sqz = squeeze_score if squeeze_score is not None else 50
                            final_growth = growth_score if growth_score is not None else 50
                            final_div = dividend_score if dividend_score is not None else 50

                            # Compute base total_score
                            # BUG FIX: Use round() instead of int() to avoid systematic downward bias
                            base_total = round(
                                final_sent * 0.25 + final_fund * 0.25 + final_tech * 0.25 +
                                final_opts * 0.15 + final_sqz * 0.10
                            )
                            # Clamp to valid range
                            base_total = max(0, min(100, base_total))

                            # Apply enhanced scoring if available
                            computed_total = base_total
                            if ENHANCED_SCORING_AVAILABLE:
                                try:
                                    enhanced_score, adjustment, _ = get_enhanced_total_score(
                                        ticker=ticker,
                                        base_score=base_total,
                                        row_data={},
                                    )
                                    computed_total = enhanced_score
                                except:
                                    pass

                            cur.execute("""
                                INSERT INTO screener_scores (
                                    ticker, date, 
                                    sentiment_score, sentiment_weighted, article_count,
                                    fundamental_score, technical_score, growth_score, dividend_score,
                                    total_score, options_flow_score, short_squeeze_score,
                                    created_at
                                ) VALUES (
                                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                                )
                            """, (
                                ticker, today,
                                _to_native(final_sent),
                                _to_native(sentiment_data.get('sentiment_weighted', final_sent)),
                                _to_native(len(articles)),
                                _to_native(final_fund),
                                _to_native(final_tech),
                                _to_native(final_growth),
                                _to_native(final_div),
                                _to_native(computed_total),
                                _to_native(final_opts),
                                _to_native(final_sqz),
                            ))

                    conn.commit()
                    logger.info(f"{ticker}: screener_scores updated for {today} - sentiment={sentiment_score}, options={options_score}")

                    # Auto-sync to historical_scores for backtesting
                    try:
                        from src.utils.historical_sync import sync_to_historical
                        sync_to_historical(ticker, today, {
                            'sentiment_score': final_sent,
                            'fundamental_score': final_fund,
                            'growth_score': final_growth,
                            'dividend_score': final_div,
                            'total_score': computed_total,
                        })
                    except Exception as e:
                        logger.debug(f"{ticker}: historical sync skipped: {e}")

            # =====================================================================
            # STEP 6: Update Earnings Calendar from yfinance
            # =====================================================================
            status_container.info(f"📅 Step 6/8: Updating earnings calendar...")

            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                ed = stock.earnings_dates

                if ed is not None and not ed.empty:
                    # Get the next upcoming earnings date
                    for idx in ed.index:
                        earnings_dt = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                        if earnings_dt >= today:
                            # Found next earnings date - save to DB
                            with get_connection() as conn:
                                with conn.cursor() as cur:
                                    try:
                                        cur.execute("""
                                            INSERT INTO earnings_calendar (ticker, earnings_date, updated_at)
                                            VALUES (%s, %s, NOW())
                                            ON CONFLICT (ticker) DO UPDATE SET
                                                earnings_date = EXCLUDED.earnings_date,
                                                updated_at = NOW()
                                        """, (ticker, earnings_dt))
                                    except Exception:
                                        # Table might not have updated_at column
                                        cur.execute("""
                                            INSERT INTO earnings_calendar (ticker, earnings_date)
                                            VALUES (%s, %s)
                                            ON CONFLICT (ticker) DO UPDATE SET
                                                earnings_date = EXCLUDED.earnings_date
                                        """, (ticker, earnings_dt))
                                conn.commit()
                            logger.info(f"{ticker}: Updated earnings_calendar -> {earnings_dt}")
                            break
            except Exception as e:
                logger.debug(f"{ticker}: Could not update earnings calendar: {e}")

            # =====================================================================
            # STEP 7: Regenerate Signal with Committee
            # =====================================================================
            status_container.info(f"🤖 Step 7/8: Regenerating signal with committee...")

            new_signal = None
            try:
                from src.core import get_signal_engine
                engine = get_signal_engine()

                # Clear this ticker from cache first
                if hasattr(engine, '_cache'):
                    for t in [ticker, ticker.upper(), ticker.lower()]:
                        if t in engine._cache:
                            del engine._cache[t]

                # Now regenerate the signal - this will reload from DB and recalculate committee
                new_signal = engine.generate_signal(ticker)
                logger.info(f"{ticker}: Regenerated signal - Today:{new_signal.today_signal} Committee:{new_signal.committee_verdict}")
            except Exception as e:
                logger.warning(f"Could not regenerate signal: {e}")

            # =====================================================================
            # STEP 8: Clear ALL caches and force refresh
            # =====================================================================
            status_container.info(f"🗑️ Step 8/8: Clearing all caches...")

            # Clear session state caches
            keys_to_clear = ['signals_data', 'market_overview']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]

            # Clear ticker-specific caches
            for key in list(st.session_state.keys()):
                if ticker in str(key) or ticker.upper() in str(key):
                    del st.session_state[key]
                    logger.info(f"Cleared session state: {key}")

            # Set flags for refresh
            st.session_state.force_refresh = True
            st.session_state.signals_loaded = True
            st.session_state['_refresh_ticker'] = ticker

            status_container.empty()

            # Show results summary
            with results_container:
                st.success(f"✅ **{ticker} Fully Updated for {today}!**")

                # Row 1: Basic metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📰 Articles", f"{len(articles)}", f"+{saved_count} new")
                with col2:
                    sent_emoji = "🟢" if sentiment_score >= 60 else "🔴" if sentiment_score <= 40 else "🟡"
                    st.metric("🧠 Sentiment", f"{sent_emoji} {sentiment_score}")
                with col3:
                    if options_score:
                        opt_emoji = "🟢" if options_score >= 60 else "🔴" if options_score <= 40 else "🟡"
                        st.metric("📊 Options", f"{opt_emoji} {options_score}")
                    else:
                        st.metric("📊 Options", "N/A")
                with col4:
                    if fundamental_score:
                        fund_emoji = "🟢" if fundamental_score >= 60 else "🔴" if fundamental_score <= 40 else "🟡"
                        st.metric("📈 Fundamental", f"{fund_emoji} {fundamental_score}")
                    else:
                        st.metric("📈 Fundamental", "N/A")

                # Row 2: Signal and Committee
                # Row 2: Signal and Committee
                # Row 2: Signal and Committee
                if new_signal:
                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        today_signal = new_signal.today_signal.value if hasattr(new_signal.today_signal,
                                                                                'value') else str(
                            new_signal.today_signal)
                        signal_emoji = "🟢" if "BUY" in str(today_signal) else "🔴" if "SELL" in str(
                            today_signal) else "🟡"
                        st.metric("📈 Today Signal", signal_emoji + " " + str(today_signal))
                    with col2:
                        verdict = str(new_signal.committee_verdict) if new_signal.committee_verdict else "N/A"
                        verdict_emoji = "🟢" if "BUY" in verdict else "🔴" if "SELL" in verdict else "🟡"
                        st.metric("🗳️ Committee", verdict_emoji + " " + verdict)
                    with col3:
                        agreement = new_signal.committee_agreement or 0
                        try:
                            agreement_num = float(agreement) if not isinstance(agreement, str) else float(
                                agreement.replace('%', '').strip())
                            st.metric("📊 Agreement", "{:.0f}%".format(agreement_num * 100))
                        except (ValueError, AttributeError):
                            st.metric("📊 Agreement", str(agreement) if agreement else "N/A")
                    with col4:
                        try:
                            score_val = int(new_signal.today_score) if new_signal.today_score else 0
                            st.metric("📊 Today Score", str(score_val))
                        except (ValueError, TypeError):
                            st.metric("📊 Today Score", str(new_signal.today_score) if new_signal.today_score else "N/A")

            # Rerun to show updated data
            time.sleep(0.3)
            st.rerun()

        except Exception as e:
            st.error(f"Error refreshing {ticker}: {e}")
            import traceback
            st.code(traceback.format_exc())