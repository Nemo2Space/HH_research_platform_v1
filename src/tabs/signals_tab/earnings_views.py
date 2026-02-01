"""
Signals Tab - Earnings Views

Earnings intelligence, news display, reaction analysis.
"""

from .shared import (
    st, pd, np, logger, datetime, date, timedelta, time, json,
    Dict, List, Optional, _to_native, text, _get_cached_reaction_analysis,
    get_engine, get_connection,
    SIGNAL_HUB_AVAILABLE, DB_AVAILABLE, REACTION_ANALYZER_AVAILABLE,
    DATEUTIL_AVAILABLE, date_parser, analyze_post_earnings,
)

def _render_news(ticker: str, earnings_focus: bool = False):
    """Render recent news with sentiment. If earnings_focus, prioritize earnings news."""
    try:
        engine = get_engine()

        # Build query - prioritize earnings news if focused
        if earnings_focus:
            # First try to get earnings-specific news
            df_earnings = pd.read_sql(f"""
                SELECT headline, ai_sentiment_fast, url, source,
                       published_at,
                       created_at,
                       COALESCE(published_at, created_at) as article_date
                FROM news_articles 
                WHERE ticker = '{ticker}' 
                AND headline IS NOT NULL AND headline != ''
                AND (
                    LOWER(headline) LIKE '%earning%'
                    OR LOWER(headline) LIKE '%quarter%'
                    OR LOWER(headline) LIKE '%revenue%'
                    OR LOWER(headline) LIKE '%profit%'
                    OR LOWER(headline) LIKE '%eps%'
                    OR LOWER(headline) LIKE '%beat%'
                    OR LOWER(headline) LIKE '%miss%'
                    OR LOWER(headline) LIKE '%guidance%'
                    OR LOWER(headline) LIKE '%forecast%'
                    OR LOWER(headline) LIKE '%results%'
                    OR LOWER(headline) LIKE '%report%'
                )
                ORDER BY COALESCE(published_at, created_at) DESC NULLS LAST
                LIMIT 8
            """, engine)

            if not df_earnings.empty:
                st.markdown("**📊 Earnings Coverage:**")
                _display_news_items(df_earnings, max_items=5)
                st.markdown("---")

        # Get all recent news
        df = pd.read_sql(f"""
            SELECT headline, ai_sentiment_fast, url, source,
                   published_at,
                   created_at,
                   COALESCE(published_at, created_at) as article_date
            FROM news_articles 
            WHERE ticker = '{ticker}' AND headline IS NOT NULL AND headline != ''
            ORDER BY COALESCE(published_at, created_at) DESC NULLS LAST
            LIMIT 15
        """, engine)

        if df.empty:
            st.caption("No recent news. Click 'Refresh' to fetch latest.")
            return

        # Filter junk
        junk = ['stock price', 'stock quote', 'google finance', 'yahoo finance', 'msn', 'marketwatch quote']
        df = df[~df['headline'].str.lower().str.contains('|'.join(junk), na=False)]

        if earnings_focus:
            st.markdown("**📰 Other Headlines:**")

        _display_news_items(df, max_items=8)

    except Exception as e:
        st.caption(f"News unavailable: {e}")


def _display_news_items(df: pd.DataFrame, max_items: int = 6):
    """Display news items with sentiment badges, source, and publication date."""

    # Invalid/generic sources to replace with URL-based source
    INVALID_SOURCES = {'brave', 'tavily', 'ai search', 'chrome', 'firefox', 'safari', 'edge', 'opera', '', 'none', 'null', 'unknown'}

    # Source name mapping from URL domains
    URL_TO_SOURCE = {
        'seekingalpha.com': 'Seeking Alpha',
        'cnbc.com': 'CNBC',
        'yahoo.com': 'Yahoo Finance',
        'finance.yahoo.com': 'Yahoo Finance',
        'bloomberg.com': 'Bloomberg',
        'reuters.com': 'Reuters',
        'wsj.com': 'WSJ',
        'ft.com': 'Financial Times',
        'marketwatch.com': 'MarketWatch',
        'fool.com': 'Motley Fool',
        'barrons.com': 'Barrons',
        'investors.com': 'IBD',
        'benzinga.com': 'Benzinga',
        'thestreet.com': 'TheStreet',
        'forbes.com': 'Forbes',
        'morningstar.com': 'Morningstar',
        'nasdaq.com': 'Nasdaq',
        'investopedia.com': 'Investopedia',
        'tipranks.com': 'TipRanks',
        'zacks.com': 'Zacks',
        'investing.com': 'Investing.com',
        'businessinsider.com': 'Business Insider',
        'nytimes.com': 'NY Times',
        'washingtonpost.com': 'Washington Post',
        'cnn.com': 'CNN',
        'bbc.com': 'BBC',
        'theguardian.com': 'Guardian',
        'marketbeat.com': 'MarketBeat',
        'simplywall.st': 'Simply Wall St',
        'tradingview.com': 'TradingView',
        'finviz.com': 'Finviz',
        'stockanalysis.com': 'Stock Analysis',
    }

    def extract_source_from_url(url: str) -> Optional[str]:
        """Extract readable source name from URL."""
        if not url:
            return None
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]

            # Check mapping first
            for url_pattern, source_name in URL_TO_SOURCE.items():
                if url_pattern in domain:
                    return source_name

            # Fallback: capitalize domain name
            # e.g., "finance.yahoo.com" -> "Yahoo" or "example.com" -> "Example"
            parts = domain.replace('.com', '').replace('.org', '').replace('.net', '').split('.')
            if parts:
                # Take the main part (usually second-to-last or last meaningful part)
                main_part = parts[-1] if len(parts) == 1 else parts[-2] if parts[-1] in ['com', 'org', 'net', 'io'] else parts[-1]
                return main_part.capitalize()
        except:
            pass
        return None

    # Filter and sort by actual publication date (newest first)
    # Only keep articles with valid published_at for proper sorting
    df_with_dates = df[df['published_at'].notna()].copy()
    df_without_dates = df[df['published_at'].isna()].copy()

    # Sort articles with dates by published_at (newest first)
    if not df_with_dates.empty:
        df_with_dates = df_with_dates.sort_values('published_at', ascending=False)

    # Combine: dated articles first (sorted by date), then undated articles
    df = pd.concat([df_with_dates, df_without_dates], ignore_index=True)

    displayed = 0
    seen_headlines = set()  # Avoid duplicates

    for _, row in df.iterrows():
        if displayed >= max_items:
            break

        headline = row.get('headline', '') or row.get('title', '')
        if not headline or len(headline) < 10:
            continue

        # Skip duplicates (similar headlines)
        headline_lower = headline.lower()[:50]
        if headline_lower in seen_headlines:
            continue
        seen_headlines.add(headline_lower)

        # Get sentiment
        sentiment = row.get('ai_sentiment_fast')
        if sentiment is not None and pd.notna(sentiment):
            try:
                sent_val = float(sentiment)
                if sent_val >= 60:
                    emoji = "🟢"
                elif sent_val <= 40:
                    emoji = "🔴"
                else:
                    emoji = "🟡"
            except:
                emoji = "🟡"
        else:
            emoji = "🟡"

        # Get source - extract from URL if source is invalid/generic
        source = str(row.get('source', '')).strip() if row.get('source') else ''
        url = row.get('url', '')

        # If source is a search engine name or empty, extract from URL
        if not source or source.lower() in INVALID_SOURCES:
            extracted = extract_source_from_url(url)
            if extracted:
                source = extracted
            else:
                source = ""  # Don't show any source if we can't determine it

        # Truncate long source names
        if source and len(source) > 25:
            source = source[:25]

        # Get date - ONLY use published_at (actual publication date)
        # Do NOT fall back to created_at/fetched_at as those show when WE saved it
        date_str = ""
        date_value = row.get('published_at')

        if date_value is not None and pd.notna(date_value):
            try:
                dt_parsed = None

                # Method 1: If already a datetime object
                if isinstance(date_value, datetime):
                    dt_parsed = date_value
                elif isinstance(date_value, pd.Timestamp):
                    dt_parsed = date_value.to_pydatetime()
                else:
                    # Method 2: Try pandas first (handles ISO format well)
                    try:
                        dt_parsed = pd.to_datetime(date_value)
                        if isinstance(dt_parsed, pd.Timestamp):
                            dt_parsed = dt_parsed.to_pydatetime()
                    except:
                        pass

                    # Method 3: Use dateutil parser
                    if dt_parsed is None and DATEUTIL_AVAILABLE and date_parser:
                        try:
                            dt_parsed = date_parser.parse(str(date_value), fuzzy=True)
                        except:
                            pass

                # Format as actual date/time in Zurich timezone
                if dt_parsed is not None:
                    try:
                        import pytz
                        zurich_tz = pytz.timezone('Europe/Zurich')

                        if dt_parsed.tzinfo is not None:
                            dt_zurich = dt_parsed.astimezone(zurich_tz)
                        else:
                            utc_tz = pytz.UTC
                            dt_utc = utc_tz.localize(dt_parsed)
                            dt_zurich = dt_utc.astimezone(zurich_tz)

                        date_str = dt_zurich.strftime("%d.%m.%Y %H:%M")
                    except:
                        if hasattr(dt_parsed, 'tzinfo') and dt_parsed.tzinfo is not None:
                            dt_parsed = dt_parsed.replace(tzinfo=None)
                        date_str = dt_parsed.strftime("%d.%m.%Y %H:%M")

            except Exception as e:
                logger.debug(f"Date parsing error for '{date_value}': {e}")

        # Build info line - show source and date if available
        info_parts = []
        if source:
            info_parts.append(source)
        if date_str:
            info_parts.append(date_str)
        info_str = " • ".join(info_parts) if info_parts else ""

        # Display: emoji + headline (as link) + source/date
        if url:
            if info_str:
                st.markdown(f"{emoji} [{headline}]({url}) — *{info_str}*")
            else:
                st.markdown(f"{emoji} [{headline}]({url})")
        else:
            if info_str:
                st.markdown(f"{emoji} {headline} — *{info_str}*")
            else:
                st.markdown(f"{emoji} {headline}")

        displayed += 1


def _render_earnings_intelligence(ei):
    """Render Earnings Intelligence section (IES/ECS) - handles both pre and post earnings."""

    if ei.is_post_earnings:
        # POST-EARNINGS VIEW
        _render_post_earnings(ei)
    else:
        # PRE-EARNINGS VIEW
        _render_pre_earnings(ei)


def _render_post_earnings(ei):
    """Render post-earnings analysis with actual results and reaction analysis."""

    days_since = abs(ei.days_to_earnings)
    st.markdown(f"**📅 Earnings {days_since} day(s) ago**")

    # ECS Result - big and prominent
    ecs_val = ei.ecs_category.value if hasattr(ei.ecs_category, 'value') else str(ei.ecs_category)

    ecs_display = {
        'STRONG_BEAT': ('🚀', 'green', 'CRUSHED IT'),
        'BEAT': ('✅', 'green', 'BEAT'),
        'INLINE': ('➡️', 'orange', 'INLINE'),
        'MISS': ('❌', 'red', 'MISSED'),
        'STRONG_MISS': ('💥', 'red', 'BADLY MISSED'),
        'PENDING': ('⏳', 'gray', 'PENDING'),
    }

    emoji, color, label = ecs_display.get(ecs_val, ('❓', 'gray', ecs_val))

    if color == 'green':
        st.success(f"{emoji} **{label}** expectations")
    elif color == 'red':
        st.error(f"{emoji} **{label}** expectations")
    else:
        st.warning(f"{emoji} **{label}** expectations")

    # EPS Result - show actual numbers
    if ei.eps_actual is not None and ei.eps_estimate is not None:
        surprise_pct = ((ei.eps_actual - ei.eps_estimate) / abs(ei.eps_estimate)) * 100 if ei.eps_estimate != 0 else 0
        eps_color = "green" if surprise_pct > 0 else "red"
        beat_miss = "BEAT" if surprise_pct > 0 else "MISS"
        st.markdown(
            f"**EPS:** ${ei.eps_actual:.2f} vs ${ei.eps_estimate:.2f} (:{eps_color}[{beat_miss} by {abs(surprise_pct):.1f}%])")
    elif hasattr(ei, 'eps_surprise_z') and ei.eps_surprise_z != 0:
        direction = "beat" if ei.eps_surprise_z > 0 else "miss"
        st.caption(f"EPS {direction} (z: {ei.eps_surprise_z:.1f})")

    # Price Reaction - THE KEY METRIC
    st.markdown("---")
    st.markdown("**📈 Market Reaction:**")

    if ei.total_reaction_pct <= -10:
        st.error(f"💥 **{ei.total_reaction_pct:+.1f}%** - CRUSHED")
    elif ei.total_reaction_pct <= -5:
        st.warning(f"📉 **{ei.total_reaction_pct:+.1f}%** - Sold Off")
    elif ei.total_reaction_pct >= 10:
        st.success(f"🚀 **{ei.total_reaction_pct:+.1f}%** - SOARED")
    elif ei.total_reaction_pct >= 5:
        st.success(f"📈 **{ei.total_reaction_pct:+.1f}%** - Rallied")
    else:
        st.info(f"➡️ **{ei.total_reaction_pct:+.1f}%** - Flat")

    # Breakdown
    if ei.gap_pct != 0 or ei.intraday_move_pct != 0:
        col1, col2 = st.columns(2)
        with col1:
            gap_emoji = "📈" if ei.gap_pct >= 0 else "📉"
            st.caption(f"{gap_emoji} Gap: {ei.gap_pct:+.1f}%")
        with col2:
            intra_emoji = "📈" if ei.intraday_move_pct >= 0 else "📉"
            st.caption(f"{intra_emoji} Intraday: {ei.intraday_move_pct:+.1f}%")

    # EQS (Earnings Quality Score)
    st.markdown("---")
    eqs_color = "green" if ei.eqs >= 60 else "red" if ei.eqs <= 40 else "orange"
    st.markdown(f"**EQS (Quality):** :{eqs_color}[{ei.eqs:.0f}/100]")
    st.progress(ei.eqs / 100)

    if ei.eqs >= 70:
        st.caption("📊 Strong earnings quality")
    elif ei.eqs <= 40:
        st.caption("📉 Weak earnings quality")

    # =========================================================================
    # REACTION ANALYZER - WHY IT MOVED + RECOMMENDATION
    # =========================================================================
    st.markdown("---")
    st.markdown("#### 🔍 Reaction Analysis")

    if REACTION_ANALYZER_AVAILABLE:
        try:
            # Get the ticker from ei
            ticker = ei.ticker if hasattr(ei, 'ticker') else None

            if ticker:
                # Use cached version to avoid re-analyzing every time
                reaction = _get_cached_reaction_analysis(ticker)

                if reaction is None:
                    st.warning("Could not analyze reaction")
                else:
                    # WHY IT MOVED
                    st.markdown("**Why did it move?**")
                    st.markdown(f"📌 **{reaction.primary_reason}**")

                if reaction.drop_reasons:
                    with st.expander("All factors", expanded=False):
                        for reason in reaction.drop_reasons[:5]:
                            st.markdown(f"• {reason}")

                # QUANTITATIVE ASSESSMENT
                st.markdown("---")
                col1, col2, col3 = st.columns(3)

                with col1:
                    # Oversold score
                    score = reaction.oversold_score
                    if score >= 65:
                        st.metric("Oversold Score", f"🟢 {score:.0f}/100", "Oversold")
                    elif score <= 40:
                        st.metric("Oversold Score", f"🟡 {score:.0f}/100", "Fair value")
                    else:
                        st.metric("Oversold Score", f"⚪ {score:.0f}/100", "Neutral")

                with col2:
                    # Assessment
                    assessment = reaction.reaction_assessment.value.replace("_", " ")
                    st.metric("Assessment", assessment)

                with col3:
                    # Confidence
                    st.metric("Confidence", f"{reaction.confidence:.0f}%")

                # RECOMMENDATION
                st.markdown("---")
                rec = reaction.recommendation.value
                rec_colors = {
                    'STRONG_BUY': ('🟢🟢', 'success'),
                    'BUY_DIP': ('🟢', 'success'),
                    'NIBBLE': ('🟢', 'info'),
                    'WAIT': ('🟡', 'warning'),
                    'AVOID': ('🔴', 'error'),
                    'SELL': ('🔴🔴', 'error'),
                    'TAKE_PROFITS': ('🟠', 'warning'),
                }
                rec_emoji, rec_type = rec_colors.get(rec, ('⚪', 'info'))

                if rec_type == 'success':
                    st.success(f"{rec_emoji} **RECOMMENDATION: {rec}**")
                elif rec_type == 'error':
                    st.error(f"{rec_emoji} **RECOMMENDATION: {rec}**")
                elif rec_type == 'warning':
                    st.warning(f"{rec_emoji} **RECOMMENDATION: {rec}**")
                else:
                    st.info(f"{rec_emoji} **RECOMMENDATION: {rec}**")

                st.caption(reaction.recommendation_reason)

                # TRADING LEVELS (if BUY recommendation)
                if rec in ['STRONG_BUY', 'BUY_DIP', 'NIBBLE'] and reaction.entry_price:
                    st.markdown("---")
                    st.markdown("**📊 Trading Levels:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Entry", f"${reaction.entry_price:.2f}")
                    with col2:
                        st.metric("Stop", f"${reaction.stop_loss:.2f}" if reaction.stop_loss else "-")
                    with col3:
                        st.metric("Target", f"${reaction.target_price:.2f}" if reaction.target_price else "-")
                    with col4:
                        st.metric("R/R", f"{reaction.risk_reward_ratio:.1f}:1" if reaction.risk_reward_ratio else "-")

                    if reaction.suggested_position_pct > 0:
                        st.caption(f"💼 Suggested position: {reaction.suggested_position_pct:.0f}% of normal size")

                # QUANT DETAILS (expandable)
                with st.expander("📐 Quantitative Details", expanded=False):
                    qm = reaction.quant_metrics

                    if qm.implied_move_pct and qm.actual_move_pct:
                        ratio_str = f"{qm.move_vs_implied_ratio:.2f}x" if qm.move_vs_implied_ratio else ""
                        verdict = "⚠️ More than expected" if qm.move_vs_implied_ratio and qm.move_vs_implied_ratio > 1.2 else "✅ Less than expected" if qm.move_vs_implied_ratio and qm.move_vs_implied_ratio < 0.8 else "≈ As expected"
                        st.markdown(
                            f"**Options implied:** ±{abs(qm.implied_move_pct):.1f}% → Actual: {qm.actual_move_pct:+.1f}% ({ratio_str}) {verdict}")

                    if qm.rsi_14:
                        rsi_status = "🔴 OVERSOLD" if qm.rsi_14 < 30 else "🟢 OVERBOUGHT" if qm.rsi_14 > 70 else "Neutral"
                        st.markdown(f"**RSI(14):** {qm.rsi_14:.0f} ({rsi_status})")

                    if qm.reaction_percentile:
                        st.markdown(f"**Reaction percentile:** {qm.reaction_percentile:.0f}th (vs historical)")

                    if qm.sector_move_pct is not None and qm.relative_to_sector is not None:
                        st.markdown(
                            f"**Sector:** {qm.sector_move_pct:+.1f}% | Stock vs sector: {qm.relative_to_sector:+.1f}%")

                    if qm.distance_to_52w_low_pct:
                        st.markdown(f"**Distance to 52w low:** {qm.distance_to_52w_low_pct:.1f}%")

            else:
                st.caption("Ticker not available for reaction analysis")

        except Exception as e:
            logger.error(f"Reaction analysis error: {e}")
            st.caption(f"Reaction analysis unavailable: {e}")
            # Fallback to simple heuristics
            _render_simple_reaction_heuristics(ei)
    else:
        # Fallback to simple heuristics when analyzer not available
        _render_simple_reaction_heuristics(ei)

    # Risk flags
    if ei.risk_flags:
        with st.expander("📋 Risk Flags"):
            for flag in ei.risk_flags:
                st.caption(flag)


def _render_simple_reaction_heuristics(ei):
    """Simple heuristic fallback when reaction analyzer not available."""
    if ei.total_reaction_pct <= -5 and ei.eqs >= 60:
        st.warning("⚠️ **SELL THE NEWS** - Good report but stock sold off. May be oversold - watch for bounce.")
    elif ei.total_reaction_pct >= 5 and ei.eqs <= 40:
        st.warning("⚠️ **BUY THE RUMOR** - Weak report but stock rallied. May be overbought.")
    elif ei.total_reaction_pct <= -8 and ei.eqs >= 45:
        st.info("💡 **POTENTIAL OPPORTUNITY** - Decent report, big selloff. Watch for mean reversion.")
    elif ei.total_reaction_pct <= -8 and ei.eqs <= 40:
        st.error("🚫 **JUSTIFIED SELLOFF** - Bad report, bad reaction. Avoid catching falling knife.")
    elif ei.total_reaction_pct >= 5 and ei.eqs >= 70:
        st.success("✅ **CONFIRMED STRENGTH** - Good report, good reaction. Momentum play.")


def _render_full_reaction_analysis_standalone(ticker: str):
    """
    Render FULL-WIDTH centralized post-earnings reaction analysis.
    STANDALONE version - doesn't need ei object, fetches everything itself.
    """
    try:
        # Use cached version to avoid re-analyzing every time
        reaction = _get_cached_reaction_analysis(ticker)

        if reaction is None:
            st.warning("Could not analyze reaction")
            return

        # =====================================================================
        # TOP ROW: Key Metrics
        # =====================================================================
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            # Earnings Result
            result_emoji = "✅" if reaction.earnings_result == "BEAT" else "❌" if reaction.earnings_result == "MISS" else "➡️"
            st.metric("Result", f"{result_emoji} {reaction.earnings_result or 'N/A'}")

        with col2:
            # Price Reaction
            if reaction.reaction_pct:
                react_emoji = "📈" if reaction.reaction_pct >= 0 else "📉"
                st.metric("Reaction", f"{react_emoji} {reaction.reaction_pct:+.1f}%")
            else:
                st.metric("Reaction", "N/A")

        with col3:
            # Oversold Score
            score = reaction.oversold_score
            score_emoji = "🟢" if score >= 65 else "🟡" if score >= 45 else "🔴"
            st.metric("Oversold Score", f"{score_emoji} {score:.0f}/100")

        with col4:
            # Assessment
            assessment = reaction.reaction_assessment.value.replace("_", " ").title()
            st.metric("Assessment", assessment)

        with col5:
            # Recommendation
            rec = reaction.recommendation.value
            rec_emoji = {'STRONG_BUY': '🟢🟢', 'BUY_DIP': '🟢', 'NIBBLE': '🟡', 'WAIT': '🟡', 'AVOID': '🔴',
                         'SELL': '🔴🔴'}.get(rec, '⚪')
            st.metric("Action", f"{rec_emoji} {rec}")

        st.markdown("---")

        # =====================================================================
        # MAIN SECTION: Two columns - Why + Recommendation
        # =====================================================================
        col_left, col_right = st.columns([3, 2])

        with col_left:
            # WHY DID IT MOVE
            st.markdown("### 🔍 Why Did It Move?")

            st.markdown(f"**Primary Reason:** {reaction.primary_reason}")

            if reaction.drop_reasons:
                st.markdown("**All Contributing Factors:**")
                for i, reason in enumerate(reaction.drop_reasons[:6], 1):
                    st.markdown(f"{i}. {reason}")

            # Key Headlines
            if reaction.key_headlines:
                st.markdown("---")
                st.markdown("**📰 Key Headlines:**")
                for headline in reaction.key_headlines[:4]:
                    st.caption(f"• {headline[:100]}...")

        with col_right:
            # RECOMMENDATION BOX
            st.markdown("### 🎯 Recommendation")

            rec = reaction.recommendation.value
            rec_configs = {
                'STRONG_BUY': ('🟢🟢', 'success', 'Strong Buy - High conviction entry'),
                'BUY_DIP': ('🟢', 'success', 'Buy the Dip - Good entry point'),
                'NIBBLE': ('🟡', 'info', 'Nibble - Small position, wait for confirmation'),
                'WAIT': ('🟡', 'warning', 'Wait - Need more clarity'),
                'AVOID': ('🔴', 'error', 'Avoid - More downside likely'),
                'SELL': ('🔴🔴', 'error', 'Sell - Exit position'),
                'TAKE_PROFITS': ('🟠', 'warning', 'Take Profits - Reduce exposure'),
            }
            rec_emoji, rec_type, rec_label = rec_configs.get(rec, ('⚪', 'info', rec))

            if rec_type == 'success':
                st.success(f"**{rec_emoji} {rec_label}**")
            elif rec_type == 'error':
                st.error(f"**{rec_emoji} {rec_label}**")
            elif rec_type == 'warning':
                st.warning(f"**{rec_emoji} {rec_label}**")
            else:
                st.info(f"**{rec_emoji} {rec_label}**")

            st.caption(f"Confidence: {reaction.confidence:.0f}%")
            st.markdown(f"*{reaction.recommendation_reason}*")

            # Trading Levels
            if rec in ['STRONG_BUY', 'BUY_DIP', 'NIBBLE'] and reaction.entry_price:
                st.markdown("---")
                st.markdown("**📊 Trading Levels:**")

                levels_col1, levels_col2 = st.columns(2)
                with levels_col1:
                    st.metric("Entry", f"${reaction.entry_price:.2f}")
                    if reaction.stop_loss:
                        st.metric("Stop Loss", f"${reaction.stop_loss:.2f}")
                with levels_col2:
                    if reaction.target_price:
                        st.metric("Target", f"${reaction.target_price:.2f}")
                    if reaction.risk_reward_ratio:
                        st.metric("Risk/Reward", f"{reaction.risk_reward_ratio:.1f}:1")

                if reaction.suggested_position_pct > 0:
                    st.info(f"💼 Suggested: {reaction.suggested_position_pct:.0f}% of normal position size")

        st.markdown("---")

        # =====================================================================
        # PRICE INFO
        # =====================================================================
        if reaction.price_before and reaction.price_current:
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                st.metric("Price Before", f"${reaction.price_before:.2f}")
            with pcol2:
                st.metric("Current Price", f"${reaction.price_current:.2f}")
            with pcol3:
                if reaction.reaction_pct:
                    change_color = "green" if reaction.reaction_pct >= 0 else "red"
                    st.metric("Change", f"{reaction.reaction_pct:+.1f}%")

        # =====================================================================
        # BOTTOM SECTION: Quantitative Details (Expandable)
        # =====================================================================
        with st.expander("📐 Quantitative Analysis Details", expanded=False):
            qcol1, qcol2, qcol3 = st.columns(3)

            qm = reaction.quant_metrics

            with qcol1:
                st.markdown("**Options Analysis:**")
                if qm.implied_move_pct:
                    st.markdown(f"• Implied move: ±{abs(qm.implied_move_pct):.1f}%")
                if qm.actual_move_pct:
                    st.markdown(f"• Actual move: {qm.actual_move_pct:+.1f}%")
                if qm.move_vs_implied_ratio:
                    verdict = "More than expected ⚠️" if qm.move_vs_implied_ratio > 1.2 else "Less than expected ✅" if qm.move_vs_implied_ratio < 0.8 else "As expected"
                    st.markdown(f"• Ratio: {qm.move_vs_implied_ratio:.2f}x ({verdict})")

            with qcol2:
                st.markdown("**Technical Analysis:**")
                if qm.rsi_14:
                    rsi_status = "OVERSOLD 🔴" if qm.rsi_14 < 30 else "OVERBOUGHT 🟢" if qm.rsi_14 > 70 else "Neutral"
                    st.markdown(f"• RSI(14): {qm.rsi_14:.0f} ({rsi_status})")
                if qm.distance_to_52w_low_pct:
                    st.markdown(f"• Distance to 52w low: {qm.distance_to_52w_low_pct:.1f}%")
                if qm.reaction_percentile:
                    st.markdown(f"• Reaction percentile: {qm.reaction_percentile:.0f}th")

            with qcol3:
                st.markdown("**Relative Performance:**")
                if qm.sector_move_pct is not None:
                    st.markdown(f"• Sector move: {qm.sector_move_pct:+.1f}%")
                if qm.relative_to_sector is not None:
                    rel_status = "Underperformed ⚠️" if qm.relative_to_sector < -3 else "Outperformed ✅" if qm.relative_to_sector > 3 else "Inline"
                    st.markdown(f"• vs Sector: {qm.relative_to_sector:+.1f}% ({rel_status})")

        # =====================================================================
        # EPS DETAILS (Expandable)
        # =====================================================================
        with st.expander("📊 Earnings Details", expanded=False):
            if reaction.eps_actual is not None or reaction.eps_estimate is not None:
                ecol1, ecol2, ecol3 = st.columns(3)

                with ecol1:
                    st.markdown("**EPS:**")
                    if reaction.eps_actual is not None:
                        st.markdown(f"• Actual: ${reaction.eps_actual:.2f}")
                    if reaction.eps_estimate is not None:
                        st.markdown(f"• Estimate: ${reaction.eps_estimate:.2f}")

                with ecol2:
                    if reaction.eps_actual is not None and reaction.eps_estimate is not None and reaction.eps_estimate != 0:
                        surprise = ((reaction.eps_actual - reaction.eps_estimate) / abs(reaction.eps_estimate)) * 100
                        st.markdown(f"**Surprise:** {surprise:+.1f}%")

                with ecol3:
                    st.markdown(f"**Result:** {reaction.earnings_result or 'N/A'}")
            else:
                st.caption("EPS details not available")

    except Exception as e:
        logger.error(f"Full reaction analysis error for {ticker}: {e}")
        import traceback
        st.error(f"Could not load reaction analysis: {e}")
        st.code(traceback.format_exc())


def _render_full_reaction_analysis(ticker: str, ei):
    """
    Render FULL-WIDTH centralized post-earnings reaction analysis.
    Shows everything about the earnings reaction in one place.
    """
    try:
        # Use cached version to avoid re-analyzing every time
        reaction = _get_cached_reaction_analysis(ticker)

        if reaction is None:
            st.warning("Could not analyze reaction")
            return

        # =====================================================================
        # TOP ROW: Key Metrics
        # =====================================================================
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            # ECS Result
            ecs_val = ei.ecs_category.value if hasattr(ei.ecs_category, 'value') else str(ei.ecs_category)
            ecs_emoji = {'STRONG_BEAT': '🚀', 'BEAT': '✅', 'INLINE': '➡️', 'MISS': '❌', 'STRONG_MISS': '💥'}.get(ecs_val,
                                                                                                               '❓')
            st.metric("Expectations", f"{ecs_emoji} {ecs_val}")

        with col2:
            # Price Reaction
            reaction_pct = ei.total_reaction_pct
            react_emoji = "📈" if reaction_pct >= 0 else "📉"
            st.metric("Reaction", f"{react_emoji} {reaction_pct:+.1f}%")

        with col3:
            # Oversold Score
            score = reaction.oversold_score
            score_emoji = "🟢" if score >= 65 else "🟡" if score >= 45 else "🔴"
            st.metric("Oversold Score", f"{score_emoji} {score:.0f}/100")

        with col4:
            # Assessment
            assessment = reaction.reaction_assessment.value.replace("_", " ").title()
            st.metric("Assessment", assessment)

        with col5:
            # Recommendation
            rec = reaction.recommendation.value
            rec_emoji = {'STRONG_BUY': '🟢🟢', 'BUY_DIP': '🟢', 'NIBBLE': '🟡', 'WAIT': '🟡', 'AVOID': '🔴',
                         'SELL': '🔴🔴'}.get(rec, '⚪')
            st.metric("Action", f"{rec_emoji} {rec}")

        st.markdown("---")

        # =====================================================================
        # MAIN SECTION: Two columns - Why + Recommendation
        # =====================================================================
        col_left, col_right = st.columns([3, 2])

        with col_left:
            # WHY DID IT MOVE
            st.markdown("### 🔍 Why Did It Move?")

            st.markdown(f"**Primary Reason:** {reaction.primary_reason}")

            if reaction.drop_reasons:
                st.markdown("**All Contributing Factors:**")
                for i, reason in enumerate(reaction.drop_reasons[:6], 1):
                    st.markdown(f"{i}. {reason}")

            # Key Headlines
            if reaction.key_headlines:
                st.markdown("---")
                st.markdown("**📰 Key Headlines:**")
                for headline in reaction.key_headlines[:4]:
                    st.caption(f"• {headline[:80]}...")

        with col_right:
            # RECOMMENDATION BOX
            st.markdown("### 🎯 Recommendation")

            rec = reaction.recommendation.value
            rec_configs = {
                'STRONG_BUY': ('🟢🟢', 'success', 'Strong Buy - High conviction entry'),
                'BUY_DIP': ('🟢', 'success', 'Buy the Dip - Good entry point'),
                'NIBBLE': ('🟡', 'info', 'Nibble - Small position, wait for confirmation'),
                'WAIT': ('🟡', 'warning', 'Wait - Need more clarity'),
                'AVOID': ('🔴', 'error', 'Avoid - More downside likely'),
                'SELL': ('🔴🔴', 'error', 'Sell - Exit position'),
                'TAKE_PROFITS': ('🟠', 'warning', 'Take Profits - Reduce exposure'),
            }
            rec_emoji, rec_type, rec_label = rec_configs.get(rec, ('⚪', 'info', rec))

            if rec_type == 'success':
                st.success(f"**{rec_emoji} {rec_label}**")
            elif rec_type == 'error':
                st.error(f"**{rec_emoji} {rec_label}**")
            elif rec_type == 'warning':
                st.warning(f"**{rec_emoji} {rec_label}**")
            else:
                st.info(f"**{rec_emoji} {rec_label}**")

            st.caption(f"Confidence: {reaction.confidence:.0f}%")
            st.markdown(f"*{reaction.recommendation_reason}*")

            # Trading Levels
            if rec in ['STRONG_BUY', 'BUY_DIP', 'NIBBLE'] and reaction.entry_price:
                st.markdown("---")
                st.markdown("**📊 Trading Levels:**")

                levels_col1, levels_col2 = st.columns(2)
                with levels_col1:
                    st.metric("Entry", f"${reaction.entry_price:.2f}")
                    if reaction.stop_loss:
                        st.metric("Stop Loss", f"${reaction.stop_loss:.2f}")
                with levels_col2:
                    if reaction.target_price:
                        st.metric("Target", f"${reaction.target_price:.2f}")
                    if reaction.risk_reward_ratio:
                        st.metric("Risk/Reward", f"{reaction.risk_reward_ratio:.1f}:1")

                if reaction.suggested_position_pct > 0:
                    st.info(f"💼 Suggested: {reaction.suggested_position_pct:.0f}% of normal position size")

        st.markdown("---")

        # =====================================================================
        # BOTTOM SECTION: Quantitative Details (Expandable)
        # =====================================================================
        with st.expander("📐 Quantitative Analysis Details", expanded=False):
            qcol1, qcol2, qcol3 = st.columns(3)

            qm = reaction.quant_metrics

            with qcol1:
                st.markdown("**Options Analysis:**")
                if qm.implied_move_pct:
                    st.markdown(f"• Implied move: ±{abs(qm.implied_move_pct):.1f}%")
                if qm.actual_move_pct:
                    st.markdown(f"• Actual move: {qm.actual_move_pct:+.1f}%")
                if qm.move_vs_implied_ratio:
                    verdict = "More than expected ⚠️" if qm.move_vs_implied_ratio > 1.2 else "Less than expected ✅" if qm.move_vs_implied_ratio < 0.8 else "As expected"
                    st.markdown(f"• Ratio: {qm.move_vs_implied_ratio:.2f}x ({verdict})")

            with qcol2:
                st.markdown("**Technical Analysis:**")
                if qm.rsi_14:
                    rsi_status = "OVERSOLD 🔴" if qm.rsi_14 < 30 else "OVERBOUGHT 🟢" if qm.rsi_14 > 70 else "Neutral"
                    st.markdown(f"• RSI(14): {qm.rsi_14:.0f} ({rsi_status})")
                if qm.distance_to_52w_low_pct:
                    st.markdown(f"• Distance to 52w low: {qm.distance_to_52w_low_pct:.1f}%")
                if qm.reaction_percentile:
                    st.markdown(f"• Reaction percentile: {qm.reaction_percentile:.0f}th")

            with qcol3:
                st.markdown("**Relative Performance:**")
                if qm.sector_move_pct is not None:
                    st.markdown(f"• Sector move: {qm.sector_move_pct:+.1f}%")
                if qm.relative_to_sector is not None:
                    rel_status = "Underperformed ⚠️" if qm.relative_to_sector < -3 else "Outperformed ✅" if qm.relative_to_sector > 3 else "Inline"
                    st.markdown(f"• vs Sector: {qm.relative_to_sector:+.1f}% ({rel_status})")

        # =====================================================================
        # EARNINGS DETAILS (Expandable)
        # =====================================================================
        with st.expander("📊 Earnings Details", expanded=False):
            ecol1, ecol2, ecol3 = st.columns(3)

            with ecol1:
                st.markdown("**EPS:**")
                if ei.eps_actual is not None and ei.eps_estimate is not None:
                    surprise = ((ei.eps_actual - ei.eps_estimate) / abs(
                        ei.eps_estimate)) * 100 if ei.eps_estimate != 0 else 0
                    st.markdown(f"• Actual: ${ei.eps_actual:.2f}")
                    st.markdown(f"• Estimate: ${ei.eps_estimate:.2f}")
                    st.markdown(f"• Surprise: {surprise:+.1f}%")

            with ecol2:
                st.markdown("**Quality Scores:**")
                st.markdown(f"• EQS (Quality): {ei.eqs:.0f}/100")
                if hasattr(ei, 'ies') and ei.ies:
                    st.markdown(f"• IES (Pre-ER): {ei.ies:.0f}/100")

            with ecol3:
                st.markdown("**Price Reaction:**")
                st.markdown(f"• Total: {ei.total_reaction_pct:+.1f}%")
                if ei.gap_pct:
                    st.markdown(f"• Gap: {ei.gap_pct:+.1f}%")
                if ei.intraday_move_pct:
                    st.markdown(f"• Intraday: {ei.intraday_move_pct:+.1f}%")

    except Exception as e:
        logger.error(f"Full reaction analysis error for {ticker}: {e}")
        st.warning(f"Could not load full reaction analysis: {e}")
        # Show basic info
        st.markdown(f"**Reaction:** {ei.total_reaction_pct:+.1f}%")
        st.markdown(f"**EQS:** {ei.eqs:.0f}/100")


def _render_pre_earnings(ei):
    """Render pre-earnings IES analysis."""

    # Regime badge
    regime_colors = {
        'HYPED': ('🔥', 'red', 'High expectations priced in'),
        'FEARED': ('😰', 'blue', 'Low expectations - upside potential'),
        'VOLATILE': ('⚡', 'orange', 'High uncertainty'),
        'NORMAL': ('📊', 'gray', 'Standard expectations'),
    }

    regime_val = ei.regime.value if hasattr(ei.regime, 'value') else str(ei.regime)
    emoji, color, desc = regime_colors.get(regime_val, ('📊', 'gray', ''))

    st.markdown(f"**Regime:** {emoji} :{color}[{regime_val}]")
    st.caption(desc)

    # IES Score with gauge
    ies_color = "red" if ei.ies >= 75 else "orange" if ei.ies >= 60 else "green" if ei.ies <= 35 else "gray"
    st.markdown(f"**IES:** :{ies_color}[{ei.ies:.0f}/100]")
    st.progress(ei.ies / 100)

    # IES interpretation
    if ei.ies >= 75:
        st.caption("⚠️ Very high expectations - needs blowout to beat")
    elif ei.ies >= 60:
        st.caption("📈 Elevated expectations - strong beat required")
    elif ei.ies <= 35:
        st.caption("✅ Low expectations - easier bar to clear")
    else:
        st.caption("📊 Normal expectations")

    # Days to earnings with urgency
    if ei.days_to_earnings <= 2:
        st.error(f"🚨 EARNINGS IN {ei.days_to_earnings} DAY(S)!")
    elif ei.days_to_earnings <= 5:
        st.warning(f"⚠️ Earnings in {ei.days_to_earnings} days")
    else:
        st.info(f"📅 Earnings in {ei.days_to_earnings} days")

    # Position scaling warning
    if ei.position_scale < 1.0:
        st.warning(f"⚖️ Position scale: {ei.position_scale:.0%}")
        st.caption("Reduce position size due to earnings risk")

    # Component breakdown in expander
    with st.expander("📊 IES Components"):
        components = [
            ("Pre-ER Runup", ei.pre_earnings_runup),
            ("Implied Move", ei.implied_move_percentile),
            ("Analyst Revisions", ei.analyst_revision_momentum),
            ("Options Skew", ei.options_skew_score),
            ("News Sentiment", ei.news_sentiment_score),
            ("Beat Rate", ei.historical_beat_rate),
            ("Sector Momentum", ei.sector_momentum),
        ]
        for name, score in components:
            emoji = "🟢" if score <= 40 else "🔴" if score >= 70 else "🟡"
            st.caption(f"{emoji} {name}: {score:.0f}")

    # Risk flags
    if ei.risk_flags:
        st.markdown("**Flags:**")
        for flag in ei.risk_flags[:3]:
            st.caption(flag)


