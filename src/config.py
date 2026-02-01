"""
HH Research Platform - Centralized Configuration

This module provides a single source of truth for all platform configurations.
All settings can be overridden via environment variables in .env file.

Usage:
    from src.config import config

    # Access settings
    model = config.ai.model
    days_back = config.news.days_back

    # Check which AI provider is active
    if config.ai.provider == "openai":
        # Use OpenAI
    elif config.ai.provider == "anthropic":
        # Use Claude
    elif config.ai.provider == "qwen":
        # Use local Qwen
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from enum import Enum

# Load .env file
load_dotenv()


def _get_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    val = os.getenv(key, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')


def _get_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _get_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _get_list(key: str, default: List[str] = None) -> List[str]:
    """Get list from comma-separated environment variable."""
    val = os.getenv(key)
    if val:
        return [x.strip() for x in val.split(',') if x.strip()]
    return default or []


class AIProvider(Enum):
    """Supported AI providers."""
    QWEN = "qwen"          # Local Qwen via llama.cpp
    OPENAI = "openai"      # OpenAI API (GPT-4, etc.)
    ANTHROPIC = "anthropic"  # Anthropic API (Claude)
    OLLAMA = "ollama"      # Ollama local models
    CUSTOM = "custom"      # Custom OpenAI-compatible endpoint


@dataclass
class AIConfig:
    """AI/LLM Configuration."""

    # Provider selection
    provider: str = os.getenv("AI_PROVIDER", "qwen")

    # Local Qwen (llama.cpp server)
    qwen_base_url: str = os.getenv("LLM_QWEN_BASE_URL", "http://localhost:8090/v1")
    qwen_model: str = os.getenv("LLM_QWEN_MODEL", "Qwen3-32B-Q6_K.gguf")

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Anthropic (Claude)
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # Ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:70b")

    # Custom endpoint (OpenAI-compatible)
    custom_base_url: str = os.getenv("CUSTOM_AI_BASE_URL", "")
    custom_model: str = os.getenv("CUSTOM_AI_MODEL", "")
    custom_api_key: str = os.getenv("CUSTOM_AI_API_KEY", "")

    # Generation parameters
    temperature: float = _get_float("AI_TEMPERATURE", 0.15)
    max_tokens: int = _get_int("AI_MAX_TOKENS", 3000)
    top_p: float = _get_float("AI_TOP_P", 0.85)
    repeat_penalty: float = _get_float("AI_REPEAT_PENALTY", 1.1)

    # Timeouts
    timeout_seconds: int = _get_int("AI_TIMEOUT", 120)

    @property
    def base_url(self) -> str:
        """Get base URL for current provider."""
        provider_map = {
            "qwen": self.qwen_base_url,
            "openai": self.openai_base_url,
            "anthropic": "https://api.anthropic.com",
            "ollama": f"{self.ollama_base_url}/v1",
            "custom": self.custom_base_url,
        }
        return provider_map.get(self.provider, self.qwen_base_url)

    @property
    def model(self) -> str:
        """Get model name for current provider."""
        provider_map = {
            "qwen": self.qwen_model,
            "openai": self.openai_model,
            "anthropic": self.anthropic_model,
            "ollama": self.ollama_model,
            "custom": self.custom_model,
        }
        return provider_map.get(self.provider, self.qwen_model)

    @property
    def api_key(self) -> Optional[str]:
        """Get API key for current provider."""
        provider_map = {
            "qwen": None,  # Local, no key needed
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "ollama": None,  # Local, no key needed
            "custom": self.custom_api_key,
        }
        return provider_map.get(self.provider)


@dataclass
class NewsConfig:
    """News collection configuration."""

    # Data collection
    days_back: int = _get_int("NEWS_DAYS_BACK", 7)
    max_articles_per_ticker: int = _get_int("NEWS_MAX_ARTICLES", 50)
    min_relevance_score: float = _get_float("NEWS_MIN_RELEVANCE", 0.5)

    # Sources
    use_finnhub: bool = _get_bool("NEWS_USE_FINNHUB", True)
    use_newsapi: bool = _get_bool("NEWS_USE_NEWSAPI", True)
    use_reddit: bool = _get_bool("NEWS_USE_REDDIT", True)
    use_google_news: bool = _get_bool("NEWS_USE_GOOGLE", True)

    # API Keys (from .env)
    finnhub_api_key: str = os.getenv("FINNHUB_API_KEY", "")
    newsapi_api_key: str = os.getenv("NEWSAPI_API_KEY", "")
    reddit_client_id: str = os.getenv("REDDIT_CLIENT_ID", "")
    reddit_client_secret: str = os.getenv("REDDIT_CLIENT_SECRET", "")

    # Cache
    cache_ttl_minutes: int = _get_int("NEWS_CACHE_TTL", 30)

    # Sentiment analysis
    use_ai_sentiment: bool = _get_bool("NEWS_USE_AI_SENTIMENT", True)
    sentiment_batch_size: int = _get_int("NEWS_SENTIMENT_BATCH", 10)


@dataclass
class OptionsConfig:
    """Options flow configuration."""

    # Data settings
    max_expiry_days: int = _get_int("OPTIONS_MAX_EXPIRY_DAYS", 45)
    min_volume_threshold: int = _get_int("OPTIONS_MIN_VOLUME", 100)
    min_oi_threshold: int = _get_int("OPTIONS_MIN_OI", 500)

    # Max pain settings
    max_pain_warning_pct: float = _get_float("OPTIONS_MAX_PAIN_WARN_PCT", 15.0)
    max_pain_error_pct: float = _get_float("OPTIONS_MAX_PAIN_ERROR_PCT", 20.0)
    max_pain_expiry_days: int = _get_int("OPTIONS_MAX_PAIN_EXPIRY_DAYS", 7)

    # P/C ratio interpretation
    pc_very_bullish: float = _get_float("OPTIONS_PC_VERY_BULLISH", 0.5)
    pc_bullish: float = _get_float("OPTIONS_PC_BULLISH", 0.7)
    pc_bearish: float = _get_float("OPTIONS_PC_BEARISH", 1.3)
    pc_very_bearish: float = _get_float("OPTIONS_PC_VERY_BEARISH", 1.5)

    # Data freshness
    stale_hours: int = _get_int("OPTIONS_STALE_HOURS", 24)


@dataclass
class SignalConfig:
    """Signal generation configuration."""

    # Score thresholds
    buy_threshold: int = _get_int("SIGNAL_BUY_THRESHOLD", 65)
    sell_threshold: int = _get_int("SIGNAL_SELL_THRESHOLD", 35)
    strong_buy_threshold: int = _get_int("SIGNAL_STRONG_BUY", 80)
    strong_sell_threshold: int = _get_int("SIGNAL_STRONG_SELL", 20)

    # Component weights (should sum to 1.0)
    weight_sentiment: float = _get_float("SIGNAL_WEIGHT_SENTIMENT", 0.25)
    weight_technical: float = _get_float("SIGNAL_WEIGHT_TECHNICAL", 0.20)
    weight_fundamental: float = _get_float("SIGNAL_WEIGHT_FUNDAMENTAL", 0.20)
    weight_options: float = _get_float("SIGNAL_WEIGHT_OPTIONS", 0.20)
    weight_earnings: float = _get_float("SIGNAL_WEIGHT_EARNINGS", 0.15)

    # Committee settings
    committee_rounds: int = _get_int("COMMITTEE_MAX_ROUNDS", 3)
    committee_consensus_threshold: float = _get_float("COMMITTEE_CONSENSUS", 0.6)

    # ML settings
    ml_min_samples: int = _get_int("ML_MIN_SAMPLES", 30)
    ml_reliability_threshold: float = _get_float("ML_RELIABILITY_THRESHOLD", 0.55)

    # Position sizing
    default_position_size: float = _get_float("SIGNAL_DEFAULT_SIZE", 1.0)
    conflict_size_cap: float = _get_float("SIGNAL_CONFLICT_SIZE_CAP", 0.25)
    ml_blocked_size_cap: float = _get_float("SIGNAL_ML_BLOCKED_SIZE_CAP", 0.25)


@dataclass
class RiskConfig:
    """Risk management configuration."""

    # Stop loss percentages by risk level
    stop_loss_low: float = _get_float("RISK_STOP_LOW", 5.0)
    stop_loss_medium: float = _get_float("RISK_STOP_MEDIUM", 7.0)
    stop_loss_high: float = _get_float("RISK_STOP_HIGH", 10.0)
    stop_loss_extreme: float = _get_float("RISK_STOP_EXTREME", 12.0)

    # Target R/R ratios
    default_rr_ratio: float = _get_float("RISK_DEFAULT_RR", 2.5)
    min_rr_ratio: float = _get_float("RISK_MIN_RR", 1.5)

    # Position limits
    max_position_pct: float = _get_float("RISK_MAX_POSITION_PCT", 10.0)
    max_sector_pct: float = _get_float("RISK_MAX_SECTOR_PCT", 30.0)


@dataclass
class DatabaseConfig:
    """Database configuration."""

    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = _get_int("POSTGRES_PORT", 5432)
    database: str = os.getenv("POSTGRES_DB", "alpha_platform")
    user: str = os.getenv("POSTGRES_USER", "alpha")
    password: str = os.getenv("POSTGRES_PASSWORD", "")

    # Connection pool
    pool_size: int = _get_int("DB_POOL_SIZE", 5)
    max_overflow: int = _get_int("DB_MAX_OVERFLOW", 10)

    @property
    def connection_string(self) -> str:
        """Get database connection string."""
        return os.getenv(
            "DATABASE_URL",
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )


@dataclass
class UIConfig:
    """UI/Dashboard configuration."""

    # Streamlit
    port: int = _get_int("STREAMLIT_PORT", 8501)

    # Display settings
    default_chart_period: str = os.getenv("UI_DEFAULT_CHART_PERIOD", "1Y")
    chart_periods: List[str] = field(default_factory=lambda: _get_list(
        "UI_CHART_PERIODS", ["1D", "5D", "1M", "3M", "6M", "1Y", "2Y", "5Y", "MAX"]
    ))

    # Table settings
    default_page_size: int = _get_int("UI_PAGE_SIZE", 25)

    # Cache
    cache_ttl_seconds: int = _get_int("UI_CACHE_TTL", 300)


@dataclass
class AlertConfig:
    """Alert/Notification configuration."""

    # Telegram
    telegram_enabled: bool = _get_bool("TELEGRAM_ENABLED", True)
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Alert thresholds
    alert_price_change_pct: float = _get_float("ALERT_PRICE_CHANGE", 5.0)
    alert_score_change: int = _get_int("ALERT_SCORE_CHANGE", 20)


@dataclass
class Config:
    """Master configuration - aggregates all config sections."""

    ai: AIConfig = field(default_factory=AIConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    options: OptionsConfig = field(default_factory=OptionsConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)

    # Application settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    universe_size: int = _get_int("UNIVERSE_SIZE", 100)
    screener_batch_size: int = _get_int("SCREENER_BATCH_SIZE", 10)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (for logging/debugging)."""
        import dataclasses
        result = {}
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if dataclasses.is_dataclass(val):
                result[f.name] = dataclasses.asdict(val)
            else:
                result[f.name] = val
        return result

    def print_summary(self) -> None:
        """Print configuration summary (hides sensitive values)."""
        print("\n" + "="*60)
        print("HH RESEARCH PLATFORM - CONFIGURATION")
        print("="*60)

        print(f"\n📡 AI Provider: {self.ai.provider.upper()}")
        print(f"   Model: {self.ai.model}")
        print(f"   Base URL: {self.ai.base_url}")
        print(f"   Temperature: {self.ai.temperature}")

        print(f"\n📰 News Settings:")
        print(f"   Days Back: {self.news.days_back}")
        print(f"   Max Articles: {self.news.max_articles_per_ticker}")
        print(f"   Sources: Finnhub={self.news.use_finnhub}, NewsAPI={self.news.use_newsapi}, Reddit={self.news.use_reddit}")

        print(f"\n📊 Options Settings:")
        print(f"   Max Pain Warning: {self.options.max_pain_warning_pct}%")
        print(f"   Max Pain Error: {self.options.max_pain_error_pct}%")
        print(f"   P/C Bullish: <{self.options.pc_bullish}, Bearish: >{self.options.pc_bearish}")

        print(f"\n🎯 Signal Settings:")
        print(f"   Buy Threshold: {self.signals.buy_threshold}")
        print(f"   Sell Threshold: {self.signals.sell_threshold}")
        print(f"   Conflict Size Cap: {self.signals.conflict_size_cap}x")

        print(f"\n⚠️ Risk Settings:")
        print(f"   Stop Loss: LOW={self.risk.stop_loss_low}%, MEDIUM={self.risk.stop_loss_medium}%, HIGH={self.risk.stop_loss_high}%")
        print(f"   Default R/R: {self.risk.default_rr_ratio}:1")

        print(f"\n🗄️ Database: {self.database.host}:{self.database.port}/{self.database.database}")

        print("="*60 + "\n")


# Global singleton instance
config = Config()


# Convenience function to reload config
def reload_config() -> Config:
    """Reload configuration from environment."""
    global config
    load_dotenv(override=True)
    config = Config()
    return config


# For backward compatibility - expose commonly used values directly
def get_ai_config() -> AIConfig:
    """Get AI configuration."""
    return config.ai


def get_news_days_back() -> int:
    """Get news days back setting."""
    return config.news.days_back


def get_max_pain_thresholds() -> tuple:
    """Get max pain warning and error thresholds."""
    return config.options.max_pain_warning_pct, config.options.max_pain_error_pct


if __name__ == "__main__":
    # Test configuration loading
    config.print_summary()