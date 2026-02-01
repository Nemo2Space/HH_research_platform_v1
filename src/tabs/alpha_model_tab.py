"""
Multi-Factor Alpha Model - Streamlit Dashboard Tab

Provides visualization and interaction with the multi-factor alpha model:
- Model training and validation metrics
- Live predictions with confidence intervals
- Factor analysis and importance
- Regime and sector breakdowns

Location: src/tabs/alpha_model_tab.py

Author: HH Research Platform
Date: December 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List
import os

# Import the alpha model
try:
    from src.ml.multi_factor_alpha import (
        MultiFactorAlphaModel,
        load_alpha_model,
        train_alpha_model,
        AlphaPrediction,
        FACTOR_DEFINITIONS,
        MarketRegime,
        SectorGroup
    )
    ALPHA_MODEL_AVAILABLE = True
except ImportError:
    ALPHA_MODEL_AVAILABLE = False

try:
    from src.utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

MODEL_PATH = 'models/multi_factor_alpha.pkl'


def render_alpha_model_tab():
    """Render the Multi-Factor Alpha Model tab."""

    st.header("🧠 Multi-Factor Alpha Model")

    if not ALPHA_MODEL_AVAILABLE:
        st.error("Multi-Factor Alpha Model not available. Please check installation.")
        st.code("pip install scikit-learn scipy xgboost")
        return

    # Check if model exists
    model_exists = os.path.exists(MODEL_PATH)

    # Sidebar controls
    with st.sidebar:
        st.subheader("🎛️ Alpha Model Controls")

        if model_exists:
            st.success("✅ Model trained and ready")
            if st.button("🔄 Retrain Model"):
                st.session_state['retrain_model'] = True
        else:
            st.warning("⚠️ No trained model found")
            if st.button("🚀 Train Model"):
                st.session_state['train_model'] = True

    # Handle training
    if st.session_state.get('train_model') or st.session_state.get('retrain_model'):
        _train_model_ui()
        st.session_state['train_model'] = False
        st.session_state['retrain_model'] = False
        st.rerun()

    if not model_exists:
        st.info("👆 Click 'Train Model' in the sidebar to get started")
        _show_model_explanation()
        return

    # Load model
    try:
        model = load_alpha_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        if st.button("🔧 Retrain Model"):
            _train_model_ui()
        return

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Live Predictions",
        "📊 Factor Analysis",
        "🎯 Model Performance",
        "🔬 Single Stock Analysis",
        "📖 Documentation"
    ])

    with tab1:
        _render_live_predictions(model)

    with tab2:
        _render_factor_analysis(model)

    with tab3:
        _render_model_performance(model)

    with tab4:
        _render_single_stock_analysis(model)

    with tab5:
        _show_model_explanation()


def _train_model_ui():
    """UI for training the model."""
    st.subheader("🚀 Training Multi-Factor Alpha Model")

    with st.spinner("Training model with walk-forward validation..."):
        progress = st.progress(0)
        status = st.empty()

        try:
            status.text("Loading historical data...")
            progress.progress(20)

            status.text("Running walk-forward validation...")
            progress.progress(40)

            # Train model
            report = train_alpha_model(
                min_date='2023-01-01',
                save_path=MODEL_PATH
            )

            progress.progress(80)
            status.text("Saving model...")

            progress.progress(100)
            status.text("Complete!")

            # Show results
            st.success("✅ Model trained successfully!")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Information Coefficient", f"{report.overall_ic:.4f}")
            with col2:
                st.metric("ICIR", f"{report.overall_icir:.2f}")
            with col3:
                st.metric("R-squared", f"{report.overall_r2:.4f}")
            with col4:
                st.metric("Beats Baseline", "✅" if report.beats_baseline else "❌")

        except Exception as e:
            st.error(f"Training failed: {e}")
            logger.error(f"Model training error: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_live_predictions(model: MultiFactorAlphaModel):
    """Render live predictions."""
    st.subheader("📈 Live Alpha Predictions")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        signal_filter = st.selectbox(
            "Filter by Signal",
            ["All", "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
        )

    with col2:
        min_conviction = st.slider("Min Conviction", 0.0, 1.0, 0.5, 0.05)

    with col3:
        top_n = st.slider("Show Top N", 10, 100, 30)

    with st.spinner("Generating predictions..."):
        try:
            predictions = model.predict_live()

            if predictions.empty:
                st.warning("No predictions available. Check database connection.")
                return

            # Apply filters
            if signal_filter != "All":
                predictions = predictions[predictions['signal'] == signal_filter]

            predictions = predictions[predictions['conviction'] >= min_conviction]

            # Show summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                buy_signals = len(predictions[predictions['signal'].isin(['STRONG_BUY', 'BUY'])])
                st.metric("🟢 Buy Signals", buy_signals)

            with col2:
                sell_signals = len(predictions[predictions['signal'].isin(['STRONG_SELL', 'SELL'])])
                st.metric("🔴 Sell Signals", sell_signals)

            with col3:
                avg_return = predictions['expected_return_5d'].mean() * 100
                st.metric("Avg Expected Return (5d)", f"{avg_return:+.2f}%")

            with col4:
                high_conf = len(predictions[predictions['confidence'] == 'HIGH'])
                st.metric("High Confidence", high_conf)

            st.markdown("---")

            # Show top predictions
            st.subheader("🎯 Top Predictions")

            # Format for display
            display_df = predictions.head(top_n).copy()
            display_df['expected_return_5d'] = display_df['expected_return_5d'].apply(lambda x: f"{x*100:+.2f}%")
            display_df['expected_return_10d'] = display_df['expected_return_10d'].apply(lambda x: f"{x*100:+.2f}%")
            display_df['ci_lower_5d'] = display_df['ci_lower_5d'].apply(lambda x: f"{x*100:+.2f}%")
            display_df['ci_upper_5d'] = display_df['ci_upper_5d'].apply(lambda x: f"{x*100:+.2f}%")
            display_df['prob_positive_5d'] = display_df['prob_positive_5d'].apply(lambda x: f"{x*100:.1f}%")
            display_df['conviction'] = display_df['conviction'].apply(lambda x: f"{x:.2f}")
            display_df['position_size'] = display_df['position_size'].apply(lambda x: f"{x:.2f}x")

            st.dataframe(
                display_df[['ticker', 'signal', 'confidence', 'expected_return_5d',
                           'ci_lower_5d', 'ci_upper_5d', 'prob_positive_5d',
                           'conviction', 'position_size', 'sector', 'top_factor']],
                use_container_width=True,
                hide_index=True
            )

            # Visualization
            st.subheader("📊 Prediction Distribution")

            # Expected return distribution
            fig = px.histogram(
                predictions,
                x=predictions['expected_return_5d'] * 100,
                nbins=30,
                title="Distribution of Expected 5-Day Returns",
                labels={'x': 'Expected Return (%)', 'y': 'Count'},
                color_discrete_sequence=['steelblue']
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero")
            st.plotly_chart(fig, use_container_width=True)

            # Signal distribution by sector
            fig2 = px.sunburst(
                predictions,
                path=['sector', 'signal'],
                title="Signals by Sector"
            )
            st.plotly_chart(fig2, use_container_width=True)

            # =====================================================================
            # INTERPRETATION & ACTIONS SECTION
            # =====================================================================
            st.markdown("---")
            st.subheader("📋 Interpretation & Recommended Actions")

            # Calculate key metrics for interpretation
            total_predictions = len(predictions)
            buy_count = len(predictions[predictions['signal'].isin(['STRONG_BUY', 'BUY'])])
            sell_count = len(predictions[predictions['signal'].isin(['STRONG_SELL', 'SELL'])])
            hold_count = len(predictions[predictions['signal'] == 'HOLD'])
            high_conf_count = len(predictions[predictions['confidence'] == 'HIGH'])
            med_conf_count = len(predictions[predictions['confidence'] == 'MEDIUM'])
            avg_conviction = predictions['conviction'].mean()
            avg_prob_positive = predictions['prob_positive_5d'].mean()

            # Market Sentiment Assessment
            if avg_return > 1:
                sentiment = "🟢 **Bullish** - Model sees positive opportunities"
                sentiment_color = "green"
            elif avg_return > 0:
                sentiment = "🟡 **Mildly Bullish** - Slight positive bias"
                sentiment_color = "orange"
            elif avg_return > -1:
                sentiment = "🟡 **Neutral** - No clear direction"
                sentiment_color = "gray"
            else:
                sentiment = "🔴 **Bearish** - Model sees downside risk"
                sentiment_color = "red"

            # Confidence Assessment
            if high_conf_count > total_predictions * 0.2:
                conf_assessment = "✅ **High confidence environment** - Model is confident in predictions"
            elif med_conf_count > total_predictions * 0.3:
                conf_assessment = "⚠️ **Medium confidence** - Some actionable signals"
            else:
                conf_assessment = "❌ **Low confidence environment** - Model is uncertain, reduce position sizes"

            # Display interpretation
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 What The Numbers Mean")
                st.markdown(f"""
                | Metric | Value | Interpretation |
                |--------|-------|----------------|
                | **Avg Expected Return** | {avg_return:+.2f}% | {'Bullish' if avg_return > 0 else 'Bearish'} outlook for next 5 days |
                | **Buy Signals** | {buy_count} | Stocks with predicted upside |
                | **Sell Signals** | {sell_count} | Stocks with predicted downside |
                | **Hold Signals** | {hold_count} | No clear edge - stay neutral |
                | **High Confidence** | {high_conf_count} | Predictions model is most sure about |
                | **Avg Conviction** | {avg_conviction:.2f} | Overall model confidence (0-1 scale) |
                | **Avg P(Positive)** | {avg_prob_positive*100:.1f}% | Probability predictions are correct |
                """)

            with col2:
                st.markdown("### 🎯 Recommended Actions")

                # Generate specific actions based on current state
                actions = []

                if buy_count > 0 and high_conf_count > 0:
                    actions.append("✅ **Consider buying** high-conviction BUY signals")
                if sell_count > 0 and high_conf_count > 0:
                    actions.append("✅ **Consider selling/shorting** high-conviction SELL signals")
                if high_conf_count == 0:
                    actions.append("⚠️ **Wait for clearer signals** - no high confidence predictions")
                if avg_conviction < 0.5:
                    actions.append("⚠️ **Reduce position sizes** - model confidence is low")
                if hold_count == total_predictions:
                    actions.append("🔄 **No action needed** - all signals are HOLD")
                if avg_prob_positive > 0.6:
                    actions.append("📈 **Bullish bias** - favor long positions")
                elif avg_prob_positive < 0.4:
                    actions.append("📉 **Bearish bias** - favor short positions or cash")
                else:
                    actions.append("⚖️ **Balanced market** - be selective with trades")

                for action in actions:
                    st.markdown(f"- {action}")

            # Column definitions expander
            with st.expander("📖 Column Definitions (click to expand)"):
                st.markdown("""
                | Column | Definition |
                |--------|------------|
                | **ticker** | Stock symbol |
                | **signal** | STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL based on expected return |
                | **confidence** | LOW/MEDIUM/HIGH - how certain the model is |
                | **expected_return_5d** | Predicted return over next 5 trading days |
                | **ci_lower_5d / ci_upper_5d** | 95% confidence interval for the prediction |
                | **prob_positive_5d** | Probability the 5-day return will be positive |
                | **conviction** | Model's confidence score (0-1). Higher = more confident |
                | **position_size** | Recommended position multiplier (e.g., 0.5x = half normal size) |
                | **sector** | Stock's sector classification |
                | **top_factor** | The factor most influencing this prediction |
                
                ### Signal Thresholds
                | Signal | Expected Return | Probability |
                |--------|-----------------|-------------|
                | STRONG_BUY | > +1.5% | > 58% |
                | BUY | > +0.5% | > 52% |
                | HOLD | -0.5% to +0.5% | 48-52% |
                | SELL | < -0.5% | < 48% |
                | STRONG_SELL | < -1.5% | < 42% |
                """)

        except Exception as e:
            st.error(f"Error generating predictions: {e}")
            logger.error(f"Prediction error: {e}")


def _render_factor_analysis(model: MultiFactorAlphaModel):
    """Render factor analysis."""
    st.subheader("📊 Factor Analysis")

    try:
        factor_df = model.get_factor_report()

        if factor_df.empty:
            st.warning("No factor data available")
            return

        # Factor importance chart
        st.subheader("Factor Importance")

        fig = px.bar(
            factor_df.head(15),
            x='importance',
            y='factor',
            orientation='h',
            color='category',
            title="Top 15 Factors by Importance",
            labels={'importance': 'Importance', 'factor': 'Factor'},
            color_discrete_map={
                'technical': '#1f77b4',
                'fundamental': '#2ca02c',
                'sentiment': '#ff7f0e',
                'institutional': '#d62728',
                'options': '#9467bd',
                'earnings': '#8c564b',
                'unknown': '#7f7f7f'
            }
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        # Factor weights for different horizons
        st.subheader("Factor Weights by Horizon")

        col1, col2 = st.columns(2)

        with col1:
            fig_5d = px.bar(
                factor_df.head(10),
                x='factor',
                y='weight_5d',
                title="5-Day Prediction Weights",
                color='weight_5d',
                color_continuous_scale=['red', 'white', 'green'],
                color_continuous_midpoint=0
            )
            fig_5d.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_5d, use_container_width=True)

        with col2:
            fig_10d = px.bar(
                factor_df.head(10),
                x='factor',
                y='weight_10d',
                title="10-Day Prediction Weights",
                color='weight_10d',
                color_continuous_scale=['red', 'white', 'green'],
                color_continuous_midpoint=0
            )
            fig_10d.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_10d, use_container_width=True)

        # Factor table
        st.subheader("Detailed Factor Table")
        st.dataframe(factor_df, use_container_width=True, hide_index=True)

        # Sign analysis
        st.subheader("Factor Sign Analysis")

        sign_match = factor_df['sign_match'].sum()
        total = len(factor_df)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Factors with Expected Sign", f"{sign_match}/{total}")
        with col2:
            st.metric("Sign Match Rate", f"{sign_match/total*100:.1f}%")

        # =====================================================================
        # INTERPRETATION & ACTIONS SECTION
        # =====================================================================
        st.markdown("---")
        st.subheader("📋 Interpretation & Recommended Actions")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 What The Numbers Mean")
            st.markdown("""
            | Metric | Interpretation |
            |--------|----------------|
            | **Importance** | How much this factor contributes to predictions (0-1). Higher = more predictive |
            | **Weight (5d/10d)** | Factor coefficient in the model. Positive = bullish signal, Negative = bearish signal |
            | **Expected Sign** | What direction we expected (1 = positive, -1 = negative) |
            | **Actual Sign** | What the model learned from data |
            | **Sign Match** | ✓ if actual matches expected, blank if opposite |
            """)

            # Analyze top factors
            top_factor = factor_df.iloc[0] if len(factor_df) > 0 else None
            if top_factor is not None:
                st.markdown(f"""
                ### 🔍 Key Findings
                
                **Top Factor: `{top_factor['factor']}`** (Importance: {top_factor['importance']:.1%})
                - This factor drives {top_factor['importance']*100:.0f}% of all predictions
                - Weight for 5-day: {top_factor['weight_5d']:.3f} ({'bullish' if top_factor['weight_5d'] > 0 else 'bearish'} signal)
                """)

                # Check for negative weights on positive factors
                negative_weights = factor_df[(factor_df['weight_5d'] < 0) & (factor_df['expected_sign'] == 1)]
                if len(negative_weights) > 0:
                    st.warning(f"⚠️ **{len(negative_weights)} factors have unexpected negative weights** - may indicate mean reversion or overbought conditions")

        with col2:
            st.markdown("### 🎯 Recommended Actions")

            actions = []

            # Sign match analysis
            sign_rate = sign_match / total if total > 0 else 0
            if sign_rate < 0.5:
                actions.append("⚠️ **Review factor assumptions** - more than half have unexpected signs (possible mean reversion)")
            else:
                actions.append("✅ **Factors behaving as expected** - model aligns with theory")

            # Concentration analysis
            if top_factor is not None and top_factor['importance'] > 0.4:
                actions.append(f"⚠️ **High factor concentration** - `{top_factor['factor']}` dominates ({top_factor['importance']:.0%}). Consider diversifying signals")

            # Weak factors
            weak_factors = factor_df[factor_df['importance'] < 0.01]
            if len(weak_factors) > 3:
                actions.append(f"📉 **{len(weak_factors)} factors have very low importance** - consider removing to simplify model")

            # Negative weight interpretation
            if top_factor is not None and top_factor['weight_5d'] < 0:
                actions.append("🔄 **Top factor has negative weight** - high scores may indicate overbought (contrarian signal)")

            for action in actions:
                st.markdown(f"- {action}")

        # Column definitions expander
        with st.expander("📖 Column Definitions (click to expand)"):
            st.markdown("""
            | Column | Definition |
            |--------|------------|
            | **factor** | Name of the scoring factor |
            | **category** | Factor type: technical, fundamental, sentiment, institutional, options |
            | **weight_5d** | Coefficient used for 5-day return predictions |
            | **weight_10d** | Coefficient used for 10-day return predictions |
            | **importance** | Relative contribution to predictions (all factors sum to ~1) |
            | **expected_sign** | Theoretical relationship: 1 = positive (higher score → higher return), -1 = negative |
            | **actual_sign** | Learned relationship from historical data |
            | **sign_match** | Whether actual matches expected (✓ = yes) |
            
            ### Understanding Negative Weights
            A **negative weight** on a typically positive factor (like `total_score`) means:
            - **Mean reversion**: High scores tend to pull back
            - **Overbought conditions**: The market may have already priced in the positive signal
            - **Contrarian signal**: Consider the opposite of what the raw score suggests
            """)

    except Exception as e:
        st.error(f"Error in factor analysis: {e}")
        logger.error(f"Factor analysis error: {e}")


def _render_model_performance(model: MultiFactorAlphaModel):
    """Render model performance metrics."""
    st.subheader("🎯 Model Performance")

    if model.validation_report is None:
        st.warning("No validation report available. Please retrain the model.")
        return

    report = model.validation_report

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Information Coefficient",
            f"{report.overall_ic:.4f}",
            help="Correlation between predictions and actual returns. Higher is better."
        )

    with col2:
        st.metric(
            "ICIR (IC / std)",
            f"{report.overall_icir:.2f}",
            help="Risk-adjusted IC. > 0.5 is good, > 1.0 is excellent."
        )

    with col3:
        st.metric(
            "R-squared",
            f"{report.overall_r2:.4f}",
            help="Variance explained by the model."
        )

    with col4:
        st.metric(
            "Statistical Significance",
            f"t={report.ic_tstat:.2f}, p={report.ic_pvalue:.3f}",
            help="T-test for IC being significantly different from zero."
        )

    st.markdown("---")

    # Walk-forward results
    st.subheader("📈 Walk-Forward Validation Results")

    if report.fold_results:
        fold_data = []
        for fold in report.fold_results:
            fold_data.append({
                'Fold': fold.get('fold', 0),
                'Train Size': fold.get('train_size', 0),
                'Test Size': fold.get('test_size', 0),
                'IC (5d)': fold.get('ic_5d', 0),
                'IC (10d)': fold.get('ic_10d', 0),
                'MSE (5d)': fold.get('mse_5d', 0)
            })

        fold_df = pd.DataFrame(fold_data)
        st.dataframe(fold_df, use_container_width=True, hide_index=True)

        # Plot IC across folds
        fig = px.line(
            fold_df,
            x='Fold',
            y='IC (5d)',
            markers=True,
            title="Information Coefficient Across Walk-Forward Folds"
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.add_hline(y=report.mean_ic_oos, line_dash="dot", line_color="green",
                     annotation_text=f"Mean IC: {report.mean_ic_oos:.4f}")
        st.plotly_chart(fig, use_container_width=True)

    # Performance by regime
    st.subheader("📊 Performance by Market Regime")

    if report.regime_performance:
        regime_data = []
        for regime, metrics in report.regime_performance.items():
            regime_data.append({
                'Regime': regime,
                'IC': metrics.get('ic', 0),
                'R²': metrics.get('r2', 0),
                'Samples': metrics.get('n', 0)
            })

        regime_df = pd.DataFrame(regime_data)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                regime_df,
                x='Regime',
                y='IC',
                color='Regime',
                title="IC by Market Regime"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.dataframe(regime_df, use_container_width=True, hide_index=True)

    # Performance by sector
    st.subheader("🏭 Performance by Sector")

    if report.sector_performance:
        sector_data = []
        for sector, metrics in report.sector_performance.items():
            sector_data.append({
                'Sector': sector,
                'IC': metrics.get('ic', 0),
                'R²': metrics.get('r2', 0),
                'Samples': metrics.get('n', 0)
            })

        sector_df = pd.DataFrame(sector_data)

        fig = px.bar(
            sector_df,
            x='Sector',
            y='IC',
            color='Sector',
            title="IC by Sector Group"
        )
        st.plotly_chart(fig, use_container_width=True)

    # =====================================================================
    # INTERPRETATION & ACTIONS SECTION
    # =====================================================================
    st.markdown("---")
    st.subheader("📋 Interpretation & Recommended Actions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 What The Numbers Mean")

        # IC interpretation
        ic = report.overall_ic
        if ic < 0.02:
            ic_quality = "❌ Very Weak (essentially random)"
        elif ic < 0.05:
            ic_quality = "⚠️ Weak (marginal predictive power)"
        elif ic < 0.10:
            ic_quality = "✅ Moderate (useful signal)"
        elif ic < 0.15:
            ic_quality = "🌟 Good (strong signal)"
        else:
            ic_quality = "🌟🌟 Excellent (very strong)"

        # ICIR interpretation
        icir = report.overall_icir
        if icir < 0.3:
            icir_quality = "❌ Inconsistent predictions"
        elif icir < 0.5:
            icir_quality = "⚠️ Somewhat consistent"
        elif icir < 1.0:
            icir_quality = "✅ Good consistency"
        else:
            icir_quality = "🌟 Excellent consistency"

        # Statistical significance
        p_value = report.ic_pvalue
        if p_value < 0.01:
            sig_quality = "🌟 Highly significant (99% confidence)"
        elif p_value < 0.05:
            sig_quality = "✅ Significant (95% confidence)"
        elif p_value < 0.10:
            sig_quality = "⚠️ Marginally significant"
        else:
            sig_quality = "❌ Not statistically significant"

        st.markdown(f"""
        | Metric | Value | Assessment |
        |--------|-------|------------|
        | **Information Coefficient** | {ic:.4f} | {ic_quality} |
        | **ICIR** | {icir:.2f} | {icir_quality} |
        | **p-value** | {p_value:.3f} | {sig_quality} |
        | **R-squared** | {report.overall_r2:.4f} | Model explains {report.overall_r2*100:.1f}% of returns |
        """)

        # Regime-specific interpretation
        if report.regime_performance:
            st.markdown("### 🏭 Regime Analysis")
            best_regime = max(report.regime_performance.items(), key=lambda x: x[1].get('ic', 0))
            worst_regime = min(report.regime_performance.items(), key=lambda x: x[1].get('ic', 0))
            st.markdown(f"""
            - **Best regime**: {best_regime[0]} (IC: {best_regime[1].get('ic', 0):.4f})
            - **Worst regime**: {worst_regime[0]} (IC: {worst_regime[1].get('ic', 0):.4f})
            - 💡 **Tip**: Trust predictions more in {best_regime[0]} market conditions
            """)

    with col2:
        st.markdown("### 🎯 Recommended Actions")

        actions = []

        # Based on overall IC
        if ic < 0.02:
            actions.append("❌ **Model needs improvement** - IC too low for reliable trading")
            actions.append("🔧 **Consider**: More training data, different features, or regime-specific models")
        elif ic < 0.05:
            actions.append("⚠️ **Use with caution** - combine with other signals")
            actions.append("💰 **Position sizing**: Reduce bet sizes due to uncertainty")
        else:
            actions.append("✅ **Model has predictive power** - can be used for trading decisions")

        # Based on statistical significance
        if p_value > 0.10:
            actions.append("⚠️ **Not statistically significant** - results could be due to chance")
            actions.append("📊 **Need more data** to confirm model validity")
        elif p_value < 0.05:
            actions.append("✅ **Statistically significant** - model edge is real (95% confidence)")

        # Based on regime performance
        if report.regime_performance:
            regime_ics = [m.get('ic', 0) for m in report.regime_performance.values()]
            if max(regime_ics) > 0.10:
                best = max(report.regime_performance.items(), key=lambda x: x[1].get('ic', 0))
                actions.append(f"🎯 **Strong in {best[0]} regime** (IC: {best[1].get('ic', 0):.2f}) - prioritize trades in this condition")

        # Based on fold consistency
        if report.fold_results:
            fold_ics = [f.get('ic_5d', 0) for f in report.fold_results]
            negative_folds = sum(1 for ic in fold_ics if ic < 0)
            if negative_folds > 0:
                actions.append(f"⚠️ **{negative_folds} fold(s) had negative IC** - model struggles in some periods")

        for action in actions:
            st.markdown(f"- {action}")

    # Metric definitions expander
    with st.expander("📖 Metric Definitions (click to expand)"):
        st.markdown("""
        | Metric | Definition | Good Value |
        |--------|------------|------------|
        | **Information Coefficient (IC)** | Correlation between predictions and actual returns | > 0.05 |
        | **ICIR (IC Information Ratio)** | IC divided by its standard deviation (consistency measure) | > 0.5 |
        | **R-squared** | Percentage of return variance explained by the model | > 0.01 |
        | **t-statistic** | How many standard errors the IC is from zero | > 2.0 |
        | **p-value** | Probability that IC is zero (lower = more significant) | < 0.05 |
        | **Beats Baseline** | Whether model outperforms simple average prediction | Yes |
        
        ### IC Benchmarks in Quantitative Finance
        | IC Range | Quality | Typical Use |
        |----------|---------|-------------|
        | < 0.02 | Noise | Not useful |
        | 0.02 - 0.05 | Weak | Combine with other signals |
        | 0.05 - 0.10 | Moderate | Standalone weak strategy |
        | 0.10 - 0.15 | Good | Solid alpha source |
        | > 0.15 | Excellent | Rare, verify no data leakage |
        
        ### Walk-Forward Validation
        The model is trained on past data and tested on future data (like real trading).
        Each "fold" is a different time period. Consistent IC across folds = robust model.
        """)


def _render_single_stock_analysis(model: MultiFactorAlphaModel):
    """Render single stock analysis."""
    st.subheader("🔬 Single Stock Analysis")

    # Ticker input
    ticker = st.text_input("Enter Ticker", value="AAPL").upper()

    if st.button("🔍 Analyze"):
        _analyze_single_stock(model, ticker)


def _analyze_single_stock(model: MultiFactorAlphaModel, ticker: str):
    """Analyze a single stock."""
    with st.spinner(f"Analyzing {ticker}..."):
        try:
            # Load live data for this ticker
            df = model.data_loader.load_live_data([ticker])

            if df.empty:
                st.warning(f"No data found for {ticker}")
                return

            row = df.iloc[0]

            # Get prediction
            factor_values = {f: row.get(f, 50) for f in model.feature_names}

            pred = model.predict(
                ticker=ticker,
                factor_values=factor_values,
                sector=row.get('sector'),
                regime=row.get('regime')
            )

            # Display results
            st.subheader(f"📊 {ticker} Analysis")

            # Signal and conviction
            signal_colors = {
                'STRONG_BUY': '🟢🟢',
                'BUY': '🟢',
                'HOLD': '🟡',
                'SELL': '🔴',
                'STRONG_SELL': '🔴🔴'
            }

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Signal",
                    f"{signal_colors.get(pred.signal, '')} {pred.signal}",
                )

            with col2:
                st.metric("Conviction", f"{pred.conviction:.0%}")

            with col3:
                st.metric("Confidence", pred.prediction_confidence)

            with col4:
                st.metric("Position Size", f"{pred.recommended_position_size:.2f}x")

            st.markdown("---")

            # Expected returns
            st.subheader("📈 Expected Returns")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "5-Day Return",
                    f"{pred.expected_return_5d*100:+.2f}%",
                    f"CI: [{pred.ci_lower_5d*100:+.1f}%, {pred.ci_upper_5d*100:+.1f}%]"
                )

            with col2:
                st.metric(
                    "10-Day Return",
                    f"{pred.expected_return_10d*100:+.2f}%",
                    f"CI: [{pred.ci_lower_10d*100:+.1f}%, {pred.ci_upper_10d*100:+.1f}%]"
                )

            with col3:
                st.metric(
                    "20-Day Return",
                    f"{pred.expected_return_20d*100:+.2f}%"
                )

            # Probabilities
            st.subheader("📊 Probabilities")

            prob_data = pd.DataFrame({
                'Metric': ['P(Positive 5d)', 'P(Positive 10d)', 'P(Beat Market 5d)'],
                'Probability': [pred.prob_positive_5d, pred.prob_positive_10d, pred.prob_beat_market_5d]
            })

            fig = px.bar(
                prob_data,
                x='Metric',
                y='Probability',
                color='Probability',
                color_continuous_scale=['red', 'yellow', 'green'],
                title="Probability Estimates"
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)

            # Factor contributions
            st.subheader("🔍 Factor Contributions")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Top Bullish Factors:**")
                for factor, contrib in pred.top_bullish_factors[:5]:
                    st.write(f"✅ {factor}: +{contrib:.3f}")

            with col2:
                st.markdown("**Top Bearish Factors:**")
                for factor, contrib in pred.top_bearish_factors[:5]:
                    st.write(f"❌ {factor}: {contrib:.3f}")

            # Factor values bar chart
            st.subheader("📊 Current Factor Values")

            factor_data = []
            for f in model.feature_names[:15]:  # Top 15 factors
                val = factor_values.get(f, 50)
                factor_data.append({'Factor': f, 'Value': val, 'vs_Neutral': val - 50})

            factor_viz_df = pd.DataFrame(factor_data)

            fig = px.bar(
                factor_viz_df,
                x='Factor',
                y='vs_Neutral',
                color='vs_Neutral',
                color_continuous_scale=['red', 'white', 'green'],
                color_continuous_midpoint=0,
                title="Factor Values vs Neutral (50)"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            # Context
            st.subheader("📋 Context")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Market Regime:** {pred.regime}")
            with col2:
                st.info(f"**Sector:** {pred.sector}")

            # Risk metrics
            st.subheader("⚖️ Risk Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Implied Sharpe Ratio", f"{pred.sharpe_ratio_implied:.2f}")
            with col2:
                st.metric("Information Ratio", f"{pred.information_ratio:.2f}")

        except Exception as e:
            st.error(f"Error analyzing {ticker}: {e}")
            logger.error(f"Single stock analysis error: {e}")
            import traceback
            st.code(traceback.format_exc())


def _show_model_explanation():
    """Show model documentation."""
    st.subheader("📖 Multi-Factor Alpha Model Documentation")

    st.markdown("""
    ## What is the Multi-Factor Alpha Model?
    
    The Multi-Factor Alpha Model is a **quantitative prediction system** that learns optimal factor weights 
    from historical data to predict stock returns. Unlike traditional scoring systems with fixed weights,
    this model:
    
    1. **Learns from data** - Weights are optimized based on what actually predicted returns historically
    2. **Adapts to market conditions** - Different weights for bull/bear/volatile markets
    3. **Sector-specific** - Recognizes that factors work differently in Tech vs Utilities
    4. **Provides confidence intervals** - Not just "BUY" but "expected +2.3% with 65% confidence"
    
    ---
    
    ## Key Concepts
    
    ### Information Coefficient (IC)
    The correlation between predicted and actual returns. 
    - IC > 0.03: Good
    - IC > 0.05: Excellent
    - IC > 0.10: Exceptional (rare)
    
    ### ICIR (Information Coefficient Information Ratio)
    IC divided by its standard deviation. Measures consistency.
    - ICIR > 0.5: Good
    - ICIR > 1.0: Excellent
    
    ### Walk-Forward Validation
    We train on past data and test on future data, simulating real trading conditions.
    This prevents overfitting and gives realistic performance estimates.
    
    ---
    
    ## Factors Used
    
    | Category | Factors |
    |----------|---------|
    | **Technical** | RSI, MACD, Momentum, Volatility |
    | **Fundamental** | Value, Quality, Growth scores |
    | **Sentiment** | News NLP, Analyst ratings, Price targets |
    | **Institutional** | Insider buying, 13F holdings, Dark pool |
    | **Options** | Put/Call ratio, IV rank, Options flow |
    
    ---
    
    ## How to Use
    
    1. **Train the model** by clicking "Train Model" in the sidebar
    2. **View live predictions** to see current opportunities
    3. **Analyze individual stocks** for detailed breakdowns
    4. **Check factor analysis** to understand what's driving predictions
    
    ---
    
    ## Interpreting Predictions
    
    | Signal | Expected Return | Probability | Action |
    |--------|----------------|-------------|--------|
    | STRONG_BUY | > +2% | > 65% | High conviction buy |
    | BUY | > +1% | > 55% | Normal buy |
    | HOLD | -1% to +1% | 45-55% | No action |
    | SELL | < -1% | < 45% | Consider selling |
    | STRONG_SELL | < -2% | < 35% | High conviction sell |
    
    ### Position Sizing
    
    The `recommended_position_size` tells you how much to allocate relative to your normal position:
    - **2.0x**: High conviction, double your normal size
    - **1.0x**: Normal position size
    - **0.5x**: Lower conviction, half your normal size
    - **0.25x**: Minimum size, very uncertain
    
    ---
    
    ## Limitations
    
    ⚠️ **Important caveats:**
    
    - Past performance doesn't guarantee future results
    - Model assumes factors continue to work similarly to the past
    - Extreme market conditions may cause model to underperform
    - Transaction costs and slippage not fully accounted for
    - Model should be one input to your decision, not the only input
    
    ---
    
    ## Technical Details
    
    - **Algorithm**: Ridge Regression with regime/sector-specific models
    - **Validation**: 5-fold walk-forward with 5-day purge gap
    - **Horizons**: 5-day, 10-day, 20-day predictions
    - **Regularization**: L2 (Ridge) to prevent overfitting
    - **Ensemble**: Weighted average by IC from global + regime + sector models
    """)


# Export for use in main app
__all__ = ['render_alpha_model_tab']