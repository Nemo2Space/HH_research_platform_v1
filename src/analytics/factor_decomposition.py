"""
Factor Decomposition Module - Phase 1

Decomposes portfolio/backtest returns into factor exposures to identify:
- How much of your alpha is real vs. disguised factor exposure
- Market beta, size, value, momentum, sector factor contributions
- True idiosyncratic alpha after removing factor returns

This answers: "Is my signal actually good, or am I just long beta/momentum?"

Author: Alpha Research Platform
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FactorType(Enum):
    """Standard factor types."""
    MARKET = "market"  # Market beta (SPY)
    SIZE = "size"  # Small vs Large cap
    VALUE = "value"  # Cheap vs Expensive (P/E, P/B)
    MOMENTUM = "momentum"  # Winners vs Losers (12-1 month return)
    QUALITY = "quality"  # High vs Low profitability (ROE, margins)
    VOLATILITY = "volatility"  # Low vol vs High vol
    SECTOR = "sector"  # Sector tilts


@dataclass
class FactorExposure:
    """Factor exposure for a single stock."""
    ticker: str
    as_of_date: date

    # Factor scores (z-scores, typically -3 to +3)
    market_beta: float = 1.0
    size_score: float = 0.0  # Positive = small cap
    value_score: float = 0.0  # Positive = cheap (value)
    momentum_score: float = 0.0  # Positive = winner
    quality_score: float = 0.0  # Positive = high quality
    volatility_score: float = 0.0  # Positive = low vol

    # Sector
    sector: str = ""

    # Raw inputs used
    market_cap: float = 0.0
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    momentum_12m: float = 0.0
    roe: float = 0.0
    volatility_60d: float = 0.0


@dataclass
class FactorReturn:
    """Factor returns for a period."""
    period_start: date
    period_end: date

    market_return: float = 0.0  # SPY return
    size_return: float = 0.0  # Small - Large return
    value_return: float = 0.0  # Value - Growth return
    momentum_return: float = 0.0  # Winners - Losers return
    quality_return: float = 0.0  # High Q - Low Q return
    volatility_return: float = 0.0  # Low Vol - High Vol return

    # Sector returns
    sector_returns: Dict[str, float] = field(default_factory=dict)


@dataclass
class FactorAttribution:
    """
    Factor attribution for a portfolio or backtest.
    Shows how much of the return came from each factor.
    """
    total_return: float  # Total portfolio return

    # Factor contributions (in return %)
    market_contribution: float = 0.0
    size_contribution: float = 0.0
    value_contribution: float = 0.0
    momentum_contribution: float = 0.0
    quality_contribution: float = 0.0
    volatility_contribution: float = 0.0
    sector_contribution: float = 0.0

    # The residual - TRUE ALPHA
    alpha: float = 0.0

    # Factor exposures (average over period)
    avg_market_beta: float = 1.0
    avg_size_exposure: float = 0.0
    avg_value_exposure: float = 0.0
    avg_momentum_exposure: float = 0.0

    # Quality metrics
    r_squared: float = 0.0  # How much variance explained by factors
    tracking_error: float = 0.0  # Volatility of residual returns
    information_ratio: float = 0.0  # Alpha / Tracking Error

    # Warnings
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_return': round(self.total_return, 2),
            'market_contribution': round(self.market_contribution, 2),
            'size_contribution': round(self.size_contribution, 2),
            'value_contribution': round(self.value_contribution, 2),
            'momentum_contribution': round(self.momentum_contribution, 2),
            'quality_contribution': round(self.quality_contribution, 2),
            'sector_contribution': round(self.sector_contribution, 2),
            'alpha': round(self.alpha, 2),
            'r_squared': round(self.r_squared, 3),
            'information_ratio': round(self.information_ratio, 2),
            'avg_market_beta': round(self.avg_market_beta, 2),
            'warnings': self.warnings,
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "=" * 50,
            "FACTOR ATTRIBUTION SUMMARY",
            "=" * 50,
            f"Total Return: {self.total_return:+.2f}%",
            "",
            "Factor Contributions:",
            f"  Market (Beta):  {self.market_contribution:+.2f}%  (beta={self.avg_market_beta:.2f})",
            f"  Size:           {self.size_contribution:+.2f}%",
            f"  Value:          {self.value_contribution:+.2f}%",
            f"  Momentum:       {self.momentum_contribution:+.2f}%",
            f"  Quality:        {self.quality_contribution:+.2f}%",
            f"  Sector:         {self.sector_contribution:+.2f}%",
            "",
            f"TRUE ALPHA:       {self.alpha:+.2f}%",
            "",
            f"R-Squared:        {self.r_squared:.1%} (variance explained by factors)",
            f"Information Ratio: {self.information_ratio:.2f}",
            "=" * 50,
        ]

        if self.warnings:
            lines.append("⚠️ WARNINGS:")
            for w in self.warnings:
                lines.append(f"  • {w}")

        return "\n".join(lines)


class FactorDataProvider:
    """
    Fetches data needed for factor calculations.
    Uses your existing database tables.
    """

    def __init__(self):
        try:
            from src.db.connection import get_engine
            self.engine = get_engine()
            self._db_available = True
        except Exception as e:
            logger.warning(f"Database not available: {e}")
            self._db_available = False
            self.engine = None

    def get_fundamentals(self, tickers: List[str], as_of_date: date = None) -> pd.DataFrame:
        """Get latest fundamentals for tickers."""
        if not self._db_available:
            return self._get_fundamentals_yfinance(tickers)

        if as_of_date is None:
            as_of_date = date.today()

        query = """
                SELECT DISTINCT \
                ON (ticker)
                    ticker, date, market_cap, pe_ratio, pb_ratio,
                    forward_pe, peg_ratio, roe, roa, profit_margin,
                    operating_margin, gross_margin,
                    revenue_growth, earnings_growth, dividend_yield,
                    debt_to_equity, current_ratio
                FROM fundamentals
                WHERE ticker = ANY (%(tickers)s)
                  AND date <= %(as_of_date)s
                ORDER BY ticker, date DESC \
                """

        try:
            df = pd.read_sql(query, self.engine, params={
                'tickers': tickers,
                'as_of_date': as_of_date
            })

            # If DB returned few results, supplement with yfinance
            if len(df) < len(tickers) * 0.5:
                logger.info("Supplementing fundamentals from yfinance...")
                yf_df = self._get_fundamentals_yfinance(
                    [t for t in tickers if t not in df['ticker'].values]
                )
                df = pd.concat([df, yf_df], ignore_index=True)

            return df
        except Exception as e:
            logger.error(f"Error fetching fundamentals from DB: {e}")
            return self._get_fundamentals_yfinance(tickers)

    def _get_fundamentals_yfinance(self, tickers: List[str]) -> pd.DataFrame:
        """Fallback to yfinance for fundamentals."""
        try:
            import yfinance as yf

            data = []
            for ticker in tickers[:50]:  # Limit to avoid rate limits
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    if info:
                        data.append({
                            'ticker': ticker,
                            'market_cap': info.get('marketCap', 0),
                            'pe_ratio': info.get('trailingPE', 0),
                            'pb_ratio': info.get('priceToBook', 0),
                            'forward_pe': info.get('forwardPE', 0),
                            'roe': info.get('returnOnEquity', 0),
                            'profit_margin': info.get('profitMargins', 0),
                            'revenue_growth': info.get('revenueGrowth', 0),
                        })
                except Exception:
                    pass

            return pd.DataFrame(data) if data else pd.DataFrame()
        except ImportError:
            return pd.DataFrame()

    def get_prices(self, tickers: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        """Get price history for tickers."""
        if not self._db_available:
            return pd.DataFrame()

        query = """
                SELECT ticker, date, close, adj_close, volume
                FROM prices
                WHERE ticker = ANY (%(tickers)s)
                  AND date >= %(start_date)s
                  AND date <= %(end_date)s
                ORDER BY ticker, date \
                """

        try:
            df = pd.read_sql(query, self.engine, params={
                'tickers': tickers,
                'start_date': start_date,
                'end_date': end_date
            })
            return df
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            return pd.DataFrame()

    def get_sector_mapping(self, tickers: List[str]) -> Dict[str, str]:
        """Get ticker to sector mapping from universe config."""
        import os
        import csv

        mapping = {}

        # Try to load from universe.csv
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config", "universe.csv"
        )

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ticker = row.get('ticker', '')
                    sector = row.get('sector', 'Unknown')
                    if ticker:
                        mapping[ticker] = sector

        # Fill in missing with Unknown
        for ticker in tickers:
            if ticker not in mapping:
                mapping[ticker] = 'Unknown'

        return mapping

    def get_benchmark_returns(self, benchmark: str, start_date: date, end_date: date) -> pd.Series:
        """Get benchmark (SPY) returns."""
        prices = self.get_prices([benchmark], start_date, end_date)

        if prices.empty:
            # Fallback to yfinance
            try:
                import yfinance as yf
                ticker = yf.Ticker(benchmark)
                hist = ticker.history(start=start_date, end=end_date)
                if not hist.empty:
                    return hist['Close'].pct_change().dropna()
            except Exception:
                pass

            return pd.Series()

        prices = prices.sort_values('date')
        price_col = 'adj_close' if 'adj_close' in prices.columns else 'close'
        returns = prices[price_col].pct_change().dropna()
        returns.index = pd.to_datetime(prices['date'].iloc[1:].values)
        return returns


class FactorCalculator:
    """
    Calculates factor exposures and returns.
    """

    def __init__(self, data_provider: FactorDataProvider = None):
        self.data_provider = data_provider or FactorDataProvider()

    def calculate_factor_exposures(self, tickers: List[str],
                                   as_of_date: date = None) -> Dict[str, FactorExposure]:
        """
        Calculate factor exposures for a list of tickers.

        Returns z-scores for each factor (cross-sectional standardization).
        """
        if as_of_date is None:
            as_of_date = date.today()

        exposures = {}

        # Get fundamentals
        fundamentals = self.data_provider.get_fundamentals(tickers, as_of_date)

        # Get prices for momentum and volatility
        start_date = as_of_date - timedelta(days=400)  # Need ~12 months
        prices = self.data_provider.get_prices(tickers, start_date, as_of_date)

        # Get sector mapping
        sectors = self.data_provider.get_sector_mapping(tickers)

        # Calculate raw factor values
        raw_data = []

        for ticker in tickers:
            row = {'ticker': ticker}

            # Fundamentals
            fund_row = fundamentals[fundamentals['ticker'] == ticker]
            if not fund_row.empty:
                fund = fund_row.iloc[0]
                row['market_cap'] = fund.get('market_cap') or 0
                row['pe_ratio'] = fund.get('pe_ratio') or 0
                row['pb_ratio'] = fund.get('pb_ratio') or 0
                row['roe'] = fund.get('roe') or 0
                row['profit_margin'] = fund.get('profit_margin') or 0
            else:
                row['market_cap'] = 0
                row['pe_ratio'] = 0
                row['pb_ratio'] = 0
                row['roe'] = 0
                row['profit_margin'] = 0

            # Momentum (12-1 month return)
            ticker_prices = prices[prices['ticker'] == ticker].sort_values('date')
            if len(ticker_prices) >= 250:
                price_col = 'adj_close' if 'adj_close' in ticker_prices.columns else 'close'
                # 12-month return, skip last month
                p_12m = ticker_prices[price_col].iloc[-252] if len(ticker_prices) >= 252 else \
                ticker_prices[price_col].iloc[0]
                p_1m = ticker_prices[price_col].iloc[-21] if len(ticker_prices) >= 21 else \
                ticker_prices[price_col].iloc[-1]
                row['momentum_12m'] = (p_1m / p_12m - 1) * 100 if p_12m > 0 else 0
            elif len(ticker_prices) >= 60:
                price_col = 'adj_close' if 'adj_close' in ticker_prices.columns else 'close'
                row['momentum_12m'] = (ticker_prices[price_col].iloc[-1] / ticker_prices[price_col].iloc[0] - 1) * 100
            else:
                row['momentum_12m'] = 0

            # Volatility (60-day)
            if len(ticker_prices) >= 60:
                price_col = 'adj_close' if 'adj_close' in ticker_prices.columns else 'close'
                returns = ticker_prices[price_col].pct_change().dropna()
                row['volatility_60d'] = returns.tail(60).std() * np.sqrt(252) * 100  # Annualized
            else:
                row['volatility_60d'] = 20  # Default 20% vol

            row['sector'] = sectors.get(ticker, 'Unknown')
            raw_data.append(row)

        # Convert to DataFrame for z-score calculation
        df = pd.DataFrame(raw_data)

        if df.empty:
            return exposures

        # Calculate z-scores (cross-sectional)
        def zscore(series):
            mean = series.mean()
            std = series.std()
            if std == 0 or pd.isna(std):
                return pd.Series([0] * len(series), index=series.index)
            return (series - mean) / std

        # Size: negative of log market cap (small = positive)
        df['size_z'] = 0.0
        valid_mcap = df['market_cap'] > 0
        if valid_mcap.any():
            df.loc[valid_mcap, 'size_z'] = -zscore(np.log(df.loc[valid_mcap, 'market_cap']))

        # Value: negative of P/E and P/B (cheap = positive)
        df['value_z'] = 0.0
        valid_pe = (df['pe_ratio'] > 0) & (df['pe_ratio'] < 200)
        valid_pb = (df['pb_ratio'] > 0) & (df['pb_ratio'] < 50)
        if valid_pe.any():
            pe_z = -zscore(df.loc[valid_pe, 'pe_ratio'])
            df.loc[valid_pe, 'value_z'] += pe_z * 0.5
        if valid_pb.any():
            pb_z = -zscore(df.loc[valid_pb, 'pb_ratio'])
            df.loc[valid_pb, 'value_z'] += pb_z * 0.5

        # Momentum: positive = winner
        df['momentum_z'] = zscore(df['momentum_12m'].clip(-100, 200))

        # Quality: ROE and profit margin
        df['quality_z'] = 0.0
        valid_roe = df['roe'].notna() & (df['roe'] > -1) & (df['roe'] < 2)
        valid_pm = df['profit_margin'].notna() & (df['profit_margin'] > -1) & (df['profit_margin'] < 1)
        if valid_roe.any():
            df.loc[valid_roe, 'quality_z'] += zscore(df.loc[valid_roe, 'roe']) * 0.5
        if valid_pm.any():
            df.loc[valid_pm, 'quality_z'] += zscore(df.loc[valid_pm, 'profit_margin']) * 0.5

        # Volatility: negative (low vol = positive, defensive)
        df['volatility_z'] = -zscore(df['volatility_60d'].clip(5, 100))

        # Create FactorExposure objects
        for _, row in df.iterrows():
            ticker = row['ticker']
            exposures[ticker] = FactorExposure(
                ticker=ticker,
                as_of_date=as_of_date,
                market_beta=1.0,  # Will calculate separately if needed
                size_score=float(row.get('size_z', 0) or 0),
                value_score=float(row.get('value_z', 0) or 0),
                momentum_score=float(row.get('momentum_z', 0) or 0),
                quality_score=float(row.get('quality_z', 0) or 0),
                volatility_score=float(row.get('volatility_z', 0) or 0),
                sector=row.get('sector', 'Unknown'),
                market_cap=float(row.get('market_cap', 0) or 0),
                pe_ratio=float(row.get('pe_ratio', 0) or 0),
                pb_ratio=float(row.get('pb_ratio', 0) or 0),
                momentum_12m=float(row.get('momentum_12m', 0) or 0),
                roe=float(row.get('roe', 0) or 0),
                volatility_60d=float(row.get('volatility_60d', 20) or 20),
            )

        return exposures

    def calculate_market_beta(self, ticker: str, benchmark: str = 'SPY',
                              lookback_days: int = 252) -> float:
        """Calculate market beta for a single ticker."""
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days + 50)

        # Get prices
        prices = self.data_provider.get_prices([ticker, benchmark], start_date, end_date)

        if prices.empty:
            return 1.0

        # Pivot to get returns
        price_col = 'adj_close' if 'adj_close' in prices.columns else 'close'
        pivot = prices.pivot(index='date', columns='ticker', values=price_col)

        if ticker not in pivot.columns or benchmark not in pivot.columns:
            return 1.0

        returns = pivot.pct_change().dropna()

        if len(returns) < 60:
            return 1.0

        # Calculate beta using covariance
        cov = returns[ticker].cov(returns[benchmark])
        var = returns[benchmark].var()

        if var == 0:
            return 1.0

        beta = cov / var
        return float(np.clip(beta, 0.2, 3.0))  # Reasonable bounds


class FactorDecomposer:
    """
    Main class for decomposing portfolio/backtest returns into factors.
    """

    def __init__(self):
        self.data_provider = FactorDataProvider()
        self.calculator = FactorCalculator(self.data_provider)

    def decompose_backtest(self, trades: List[Dict],
                           holding_period: int = 10) -> FactorAttribution:
        """
        Decompose backtest returns into factor contributions.

        Args:
            trades: List of trade dicts with ticker, entry_date, return_pct
            holding_period: Holding period in days

        Returns:
            FactorAttribution with breakdown
        """
        if not trades:
            return FactorAttribution(total_return=0.0, alpha=0.0)

        # Get unique tickers and date range
        tickers = list(set(t.get('ticker') for t in trades))

        # Parse dates
        def parse_date(d):
            if isinstance(d, date):
                return d
            if isinstance(d, datetime):
                return d.date()
            if isinstance(d, str):
                return datetime.strptime(d[:10], '%Y-%m-%d').date()
            return date.today()

        dates = [parse_date(t.get('entry_date')) for t in trades]
        start_date = min(dates) - timedelta(days=30)
        end_date = max(dates) + timedelta(days=holding_period + 30)

        # Calculate factor exposures at midpoint
        mid_date = dates[len(dates) // 2]
        exposures = self.calculator.calculate_factor_exposures(tickers, mid_date)

        # Get market returns
        benchmark_returns = self.data_provider.get_benchmark_returns('SPY', start_date, end_date)

        # Aggregate trade returns and factor exposures
        total_return = sum(t.get('return_pct', 0) or t.get('return_pct_net', 0) for t in trades)
        n_trades = len(trades)
        avg_return = total_return / n_trades if n_trades > 0 else 0

        # Average factor exposures weighted by trades
        avg_size = 0.0
        avg_value = 0.0
        avg_momentum = 0.0
        avg_quality = 0.0
        avg_beta = 0.0

        sector_weights = {}

        for trade in trades:
            ticker = trade.get('ticker')
            exp = exposures.get(ticker)

            if exp:
                avg_size += exp.size_score
                avg_value += exp.value_score
                avg_momentum += exp.momentum_score
                avg_quality += exp.quality_score
                avg_beta += exp.market_beta

                sector = exp.sector
                sector_weights[sector] = sector_weights.get(sector, 0) + 1

        if n_trades > 0:
            avg_size /= n_trades
            avg_value /= n_trades
            avg_momentum /= n_trades
            avg_quality /= n_trades
            avg_beta /= n_trades

        # Estimate factor returns for the period
        # Using simplified factor premium estimates (annualized, scaled to holding period)
        scale = holding_period / 252  # Scale annual to holding period

        # Get actual market return
        if not benchmark_returns.empty:
            market_return = benchmark_returns.sum() * 100  # Convert to %
        else:
            market_return = 10.0 * scale  # Assume ~10% annual

        # Simplified factor premiums (annual, from academic research)
        # These are rough estimates - real implementation would calculate from data
        size_premium = 2.0 * scale  # Small-cap premium ~2% annual
        value_premium = 3.0 * scale  # Value premium ~3% annual
        momentum_premium = 6.0 * scale  # Momentum premium ~6% annual
        quality_premium = 3.0 * scale  # Quality premium ~3% annual

        # Calculate factor contributions
        # Contribution = exposure * factor_return
        market_contribution = (avg_beta - 1.0) * market_return  # Excess beta contribution
        size_contribution = avg_size * size_premium
        value_contribution = avg_value * value_premium
        momentum_contribution = avg_momentum * momentum_premium
        quality_contribution = avg_quality * quality_premium

        # Sector contribution (simplified - deviation from market sector weights)
        sector_contribution = 0.0  # Would need sector returns to calculate properly

        # Total factor contribution
        total_factor_contribution = (
                market_contribution +
                size_contribution +
                value_contribution +
                momentum_contribution +
                quality_contribution +
                sector_contribution
        )

        # Alpha = Total Return - Factor Contributions - Market Return
        # Note: Market return itself isn't alpha, only EXCESS beta is
        base_market = market_return  # What you'd get with beta=1
        alpha = total_return - base_market - total_factor_contribution

        # Estimate R-squared (how much variance explained)
        # Simplified estimate based on factor exposure magnitude
        exposure_magnitude = abs(avg_size) + abs(avg_value) + abs(avg_momentum) + abs(avg_quality)
        r_squared = min(0.9, 0.3 + exposure_magnitude * 0.1)  # Rough estimate

        # Information Ratio = Alpha / Tracking Error
        # Estimate tracking error from volatility of residuals
        tracking_error = 15.0  # Assume ~15% tracking error (would calculate from data)
        information_ratio = (alpha * (252 / holding_period)) / tracking_error if tracking_error > 0 else 0

        # Generate warnings
        warnings = []

        if avg_beta > 1.3:
            warnings.append(f"High market beta ({avg_beta:.2f}) - returns may drop in bear market")
        if avg_momentum > 1.5:
            warnings.append(f"Strong momentum tilt ({avg_momentum:.2f}) - vulnerable to reversal")
        if abs(avg_value) > 1.5:
            warnings.append(f"Strong value/growth tilt - style rotation risk")
        if market_contribution > total_return * 0.5:
            warnings.append("More than 50% of return from market beta - consider hedging")
        if alpha < 0:
            warnings.append("Negative alpha - strategy underperforming after factor adjustment")

        return FactorAttribution(
            total_return=total_return,
            market_contribution=market_contribution,
            size_contribution=size_contribution,
            value_contribution=value_contribution,
            momentum_contribution=momentum_contribution,
            quality_contribution=quality_contribution,
            volatility_contribution=0.0,
            sector_contribution=sector_contribution,
            alpha=alpha,
            avg_market_beta=avg_beta,
            avg_size_exposure=avg_size,
            avg_value_exposure=avg_value,
            avg_momentum_exposure=avg_momentum,
            r_squared=r_squared,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            warnings=warnings,
        )

    def decompose_portfolio(self, positions: List[Dict],
                            period_days: int = 30) -> FactorAttribution:
        """
        Decompose current portfolio's factor exposures.

        Args:
            positions: List of position dicts with ticker, weight, return_pct
            period_days: Period for return calculation

        Returns:
            FactorAttribution showing current factor tilts
        """
        if not positions:
            return FactorAttribution(total_return=0.0, alpha=0.0)

        tickers = [p.get('ticker') or p.get('symbol') for p in positions]
        weights = [p.get('weight', 1.0 / len(positions)) for p in positions]

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # Calculate exposures
        exposures = self.calculator.calculate_factor_exposures(tickers)

        # Weighted average exposures
        avg_beta = 0.0
        avg_size = 0.0
        avg_value = 0.0
        avg_momentum = 0.0
        avg_quality = 0.0

        for ticker, weight in zip(tickers, weights):
            exp = exposures.get(ticker)
            if exp:
                avg_beta += exp.market_beta * weight
                avg_size += exp.size_score * weight
                avg_value += exp.value_score * weight
                avg_momentum += exp.momentum_score * weight
                avg_quality += exp.quality_score * weight

        # Get portfolio return if available
        total_return = sum(
            (p.get('return_pct', 0) or 0) * w
            for p, w in zip(positions, weights)
        )

        # Generate warnings for extreme tilts
        warnings = []
        if avg_beta > 1.2:
            warnings.append(f"High beta portfolio ({avg_beta:.2f})")
        if avg_size > 1.0:
            warnings.append(f"Small-cap tilt ({avg_size:.2f})")
        if avg_momentum > 1.0:
            warnings.append(f"Momentum tilt ({avg_momentum:.2f})")

        return FactorAttribution(
            total_return=total_return,
            avg_market_beta=avg_beta,
            avg_size_exposure=avg_size,
            avg_value_exposure=avg_value,
            avg_momentum_exposure=avg_momentum,
            alpha=0.0,  # Need returns data to calculate
            warnings=warnings,
        )

    def get_factor_exposures_table(self, tickers: List[str]) -> pd.DataFrame:
        """
        Get a DataFrame of factor exposures for display.
        """
        exposures = self.calculator.calculate_factor_exposures(tickers)

        data = []
        for ticker, exp in exposures.items():
            data.append({
                'Ticker': ticker,
                'Sector': exp.sector,
                'Size': round(exp.size_score, 2),
                'Value': round(exp.value_score, 2),
                'Momentum': round(exp.momentum_score, 2),
                'Quality': round(exp.quality_score, 2),
                'Volatility': round(exp.volatility_score, 2),
                'Market Cap ($B)': round(exp.market_cap / 1e9, 1) if exp.market_cap > 0 else 0,
                'P/E': round(exp.pe_ratio, 1) if exp.pe_ratio > 0 else 'N/A',
                '12m Mom %': round(exp.momentum_12m, 1),
            })

        return pd.DataFrame(data)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_decomposer = None


def get_factor_decomposer() -> FactorDecomposer:
    """Get singleton FactorDecomposer instance."""
    global _decomposer
    if _decomposer is None:
        _decomposer = FactorDecomposer()
    return _decomposer


def decompose_backtest_returns(trades: List[Dict],
                               holding_period: int = 10) -> FactorAttribution:
    """
    Decompose backtest returns into factor contributions.

    Usage:
        result = engine.run_backtest(...)
        trades = [{'ticker': t.ticker, 'entry_date': t.entry_date,
                   'return_pct': t.return_pct} for t in result.trades]
        attribution = decompose_backtest_returns(trades)
        print(attribution.get_summary())
    """
    decomposer = get_factor_decomposer()
    return decomposer.decompose_backtest(trades, holding_period)


def analyze_portfolio_factors(positions: List[Dict]) -> FactorAttribution:
    """
    Analyze factor exposures of current portfolio.

    Usage:
        positions = [{'ticker': 'AAPL', 'weight': 0.10}, ...]
        attribution = analyze_portfolio_factors(positions)
        print(attribution.get_summary())
    """
    decomposer = get_factor_decomposer()
    return decomposer.decompose_portfolio(positions)


def get_ticker_exposures(tickers: List[str]) -> pd.DataFrame:
    """
    Get factor exposure table for a list of tickers.

    Usage:
        df = get_ticker_exposures(['AAPL', 'NVDA', 'MSFT'])
        print(df)
    """
    decomposer = get_factor_decomposer()
    return decomposer.get_factor_exposures_table(tickers)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Factor Decomposition...")

    # Test with mock trades
    mock_trades = [
        {'ticker': 'AAPL', 'entry_date': '2024-01-15', 'return_pct': 5.2},
        {'ticker': 'NVDA', 'entry_date': '2024-01-16', 'return_pct': 12.3},
        {'ticker': 'MSFT', 'entry_date': '2024-01-17', 'return_pct': 3.1},
        {'ticker': 'AMZN', 'entry_date': '2024-01-18', 'return_pct': -2.5},
        {'ticker': 'GOOGL', 'entry_date': '2024-01-19', 'return_pct': 4.7},
    ]

    decomposer = FactorDecomposer()
    attribution = decomposer.decompose_backtest(mock_trades, holding_period=10)

    print(attribution.get_summary())