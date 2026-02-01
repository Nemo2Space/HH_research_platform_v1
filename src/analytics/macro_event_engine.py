"""
Macro/Geopolitical Event Engine

Structured "Event → Macro Factors → Asset/sector mapping" layer.

Design Principles (from README):
1. DETERMINISTIC Python calculations - no LLM-invented numbers
2. EVENT-LED not headline-led - deduplicate to single events
3. SOURCE-TIERED scoring with recency decay
4. READ-ONLY integration first - produces context, doesn't override scores

Components:
1. Event Ingestion - from existing news sources (no new complexity)
2. Event Classification + Entity Extraction - rule-based + LLM-assisted labeling
3. Factor Score Computation - deterministic rubric with source tiers
4. Exposure Mapping - sector/asset sensitivities to factors

Location: src/analytics/macro_event_engine.py
Author: HH Research Platform
"""

import os
import sys
import re
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    # Add project root to path for standalone testing
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

try:
    from src.utils.logging import get_logger
    from src.db.connection import get_connection, get_engine
except ImportError:
    # Fallback for standalone testing
    import logging
    logging.basicConfig(level=logging.INFO)
    def get_logger(name):
        return logging.getLogger(name)

    def get_connection():
        raise ImportError("Database connection not available in standalone mode")

    def get_engine():
        raise ImportError("Database engine not available in standalone mode")

logger = get_logger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class EventType(Enum):
    """Macro event taxonomy."""
    # Geopolitical
    CONFLICT = "CONFLICT"              # Wars, military actions
    SANCTIONS = "SANCTIONS"            # Trade sanctions, asset freezes
    COUP_POLITICAL_CRISIS = "COUP_POLITICAL_CRISIS"  # Coups, regime changes
    DIPLOMATIC = "DIPLOMATIC"          # Summits, treaties, tensions

    # Policy
    TARIFF = "TARIFF"                  # Trade tariffs, duties
    REGULATION = "REGULATION"          # New laws, regulatory changes
    FISCAL_POLICY = "FISCAL_POLICY"   # Government spending, tax changes
    ELECTION = "ELECTION"              # Elections, political transitions

    # Energy/Commodities
    OPEC_DECISION = "OPEC_DECISION"   # OPEC+ production decisions
    ENERGY_DISRUPTION = "ENERGY_DISRUPTION"  # Pipeline attacks, outages
    COMMODITY_SHOCK = "COMMODITY_SHOCK"  # Supply/demand shocks

    # Macro Regime
    CENTRAL_BANK = "CENTRAL_BANK"     # Fed, ECB, BOJ decisions
    INFLATION_DATA = "INFLATION_DATA"  # CPI, PCE releases
    RECESSION_SIGNAL = "RECESSION_SIGNAL"  # GDP, employment shocks
    FX_CRISIS = "FX_CRISIS"           # Currency crises, devaluations

    # Market Events
    EARNINGS_SHOCK = "EARNINGS_SHOCK"  # Major earnings surprises
    DEFAULT_CREDIT = "DEFAULT_CREDIT"  # Sovereign/corporate defaults

    UNKNOWN = "UNKNOWN"


class EventSeverity(Enum):
    """Event severity/impact level."""
    CRITICAL = 5    # Market-moving, immediate action needed
    HIGH = 4        # Significant impact, close monitoring
    MEDIUM = 3      # Notable, factor into analysis
    LOW = 2         # Minor, background noise
    MINIMAL = 1     # Negligible


# Source credibility tiers (consistent with your existing SOURCE_CREDIBILITY)
SOURCE_TIERS = {
    # Tier 1: Most credible (weight 1.0)
    'reuters': 1.0, 'associated press': 1.0, 'ap news': 1.0, 'ap': 1.0,
    'bloomberg': 1.0, 'wsj': 1.0, 'wall street journal': 1.0,
    'financial times': 1.0, 'ft': 1.0, 'ft.com': 1.0,
    'the economist': 1.0,

    # Tier 2: Very credible (weight 0.85)
    'cnbc': 0.85, 'bbc': 0.85, 'bbc news': 0.85,
    'new york times': 0.85, 'nytimes': 0.85, 'washington post': 0.85,
    'guardian': 0.85, 'the guardian': 0.85,

    # Tier 3: Credible (weight 0.7)
    'marketwatch': 0.7, 'barrons': 0.7, "barron's": 0.7,
    'yahoo finance': 0.7, 'yahoo': 0.7, 'investing.com': 0.7,
    'cnn': 0.7, 'cnn business': 0.7, 'fox business': 0.7,

    # Tier 4: Moderate (weight 0.5)
    'seeking alpha': 0.5, 'benzinga': 0.5, 'zacks': 0.5,
    'motley fool': 0.5, 'thestreet': 0.5,

    # Tier 5: Low credibility (weight 0.3)
    'unknown': 0.3, 'other': 0.3,
}

# Recency decay parameters (hours)
RECENCY_DECAY = {
    'full_weight_hours': 6,      # Full weight for first 6 hours
    'half_life_hours': 24,       # Half weight after 24 hours
    'min_weight': 0.1,           # Minimum weight (10%)
    'stale_hours': 72,           # Consider stale after 72 hours
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MacroEvent:
    """A single deduplicated macro event."""
    event_id: str                      # Hash of normalized event
    event_type: EventType
    severity: EventSeverity

    # Core info
    title: str                         # Canonical event title
    summary: str                       # Brief summary
    entities: Dict[str, List[str]]     # {countries: [], commodities: [], orgs: []}

    # Timing
    first_seen: datetime
    last_updated: datetime
    is_ongoing: bool = True

    # Source tracking
    sources: List[str] = field(default_factory=list)
    source_count: int = 0
    headline_count: int = 0            # How many headlines map to this event

    # Confidence (0-100, computed from sources)
    confidence: int = 50

    # Factor impacts (which factors this event affects)
    factor_impacts: Dict[str, int] = field(default_factory=dict)  # factor_name: delta

    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'summary': self.summary,
            'entities': self.entities,
            'first_seen': self.first_seen.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'is_ongoing': self.is_ongoing,
            'sources': self.sources,
            'source_count': self.source_count,
            'headline_count': self.headline_count,
            'confidence': self.confidence,
            'factor_impacts': self.factor_impacts,
        }


@dataclass
class MacroFactorScores:
    """
    Structured macro factor outputs.
    All scores are 0-100, where 50 = neutral/baseline.
    >50 means elevated risk/impact, <50 means reduced.
    """
    timestamp: datetime = field(default_factory=datetime.now)

    # Geopolitical factors
    oil_supply_shock: int = 50         # Oil supply disruption risk
    sanctions_risk: int = 50           # Sanctions escalation
    conflict_risk: int = 50            # War/military conflict
    geopolitical_tension: int = 50     # General geopolitical stress

    # Policy factors
    trade_war_risk: int = 50           # Tariff/trade war escalation
    regulation_risk: int = 50          # Regulatory tightening
    fiscal_stimulus: int = 50          # Government spending (>50 = stimulative)
    political_uncertainty: int = 50    # Election/transition uncertainty

    # Macro regime factors
    inflation_pressure: int = 50       # Inflation risk
    recession_risk: int = 50           # Economic slowdown risk
    risk_off_sentiment: int = 50       # Flight to safety
    fx_stress: int = 50                # Currency/EM stress
    credit_stress: int = 50            # Credit/default risk

    # Commodity factors
    energy_disruption: int = 50        # Energy supply risk
    supply_chain_stress: int = 50      # Logistics/shipping disruption
    commodity_volatility: int = 50     # Broad commodity stress

    # Metadata
    active_event_count: int = 0
    data_freshness: str = "UNKNOWN"    # FRESH, STALE, NO_DATA
    last_event_time: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def get_elevated_factors(self, threshold: int = 60) -> List[Tuple[str, int]]:
        """Get factors above threshold."""
        elevated = []
        for factor in ['oil_supply_shock', 'sanctions_risk', 'conflict_risk',
                       'geopolitical_tension', 'trade_war_risk', 'regulation_risk',
                       'inflation_pressure', 'recession_risk', 'risk_off_sentiment',
                       'fx_stress', 'credit_stress', 'energy_disruption',
                       'supply_chain_stress', 'commodity_volatility', 'political_uncertainty']:
            value = getattr(self, factor, 50)
            if value >= threshold:
                elevated.append((factor, value))
        return sorted(elevated, key=lambda x: x[1], reverse=True)

    def get_reduced_factors(self, threshold: int = 40) -> List[Tuple[str, int]]:
        """Get factors below threshold (positive signals)."""
        reduced = []
        for factor in ['trade_war_risk', 'recession_risk', 'inflation_pressure']:
            value = getattr(self, factor, 50)
            if value <= threshold:
                reduced.append((factor, value))
        return sorted(reduced, key=lambda x: x[1])


@dataclass
class SectorExposure:
    """How a sector is exposed to macro factors."""
    sector: str
    factor_sensitivities: Dict[str, float]  # factor_name: sensitivity (-1 to +1)
    current_impact: float = 0.0             # Computed from factors * sensitivities
    tailwinds: List[str] = field(default_factory=list)
    headwinds: List[str] = field(default_factory=list)


@dataclass
class PortfolioMacroExposure:
    """Portfolio-level macro exposure analysis."""
    timestamp: datetime
    factors: MacroFactorScores
    sector_exposures: Dict[str, SectorExposure]
    ticker_impacts: Dict[str, float]        # ticker: net_impact

    # Aggregates
    net_portfolio_impact: float = 0.0       # -100 to +100
    top_tailwinds: List[Tuple[str, float]] = field(default_factory=list)
    top_headwinds: List[Tuple[str, float]] = field(default_factory=list)

    # Alerts
    alerts: List[str] = field(default_factory=list)


# =============================================================================
# SECTOR FACTOR SENSITIVITIES
# =============================================================================

# How each sector responds to macro factors
# Positive = benefits, Negative = hurt
# Scale: -1.0 to +1.0

SECTOR_FACTOR_SENSITIVITY = {
    'Energy': {
        'oil_supply_shock': +0.80,      # Benefits from supply shocks (higher prices)
        'sanctions_risk': +0.30,        # Often benefits from sanctions on producers
        'conflict_risk': +0.40,         # Geopolitical premium
        'inflation_pressure': +0.20,    # Inflation hedge
        'recession_risk': -0.50,        # Demand destruction
        'risk_off_sentiment': -0.30,
    },
    'Financials': {
        'inflation_pressure': +0.30,    # Higher rates help NIMs
        'recession_risk': -0.60,        # Credit losses
        'credit_stress': -0.70,         # Direct exposure
        'fx_stress': -0.20,
        'regulation_risk': -0.40,
    },
    'Technology': {
        'trade_war_risk': -0.60,        # China exposure, tariffs
        'recession_risk': -0.40,        # Discretionary spending
        'risk_off_sentiment': -0.50,    # Growth rotation
        'regulation_risk': -0.30,       # Antitrust, privacy
        'inflation_pressure': -0.20,    # Discount rate impact
    },
    'Healthcare': {
        'recession_risk': +0.20,        # Defensive
        'risk_off_sentiment': +0.15,    # Flight to safety
        'regulation_risk': -0.40,       # Drug pricing, policy
        'political_uncertainty': -0.20,
    },
    'Consumer Discretionary': {
        'recession_risk': -0.70,        # Highly cyclical
        'inflation_pressure': -0.50,    # Hurts spending
        'trade_war_risk': -0.30,        # Import costs
        'risk_off_sentiment': -0.40,
    },
    'Consumer Staples': {
        'recession_risk': +0.30,        # Defensive
        'risk_off_sentiment': +0.25,    # Flight to safety
        'inflation_pressure': -0.20,    # Input costs
        'supply_chain_stress': -0.30,
    },
    'Industrials': {
        'recession_risk': -0.60,        # Cyclical
        'trade_war_risk': -0.40,        # Global supply chains
        'supply_chain_stress': -0.50,
        'fiscal_stimulus': +0.50,       # Infrastructure spending
        'conflict_risk': +0.20,         # Defense component
    },
    'Materials': {
        'recession_risk': -0.50,        # Cyclical
        'inflation_pressure': +0.30,    # Commodity exposure
        'supply_chain_stress': -0.30,
        'fiscal_stimulus': +0.40,       # Infrastructure
    },
    'Utilities': {
        'recession_risk': +0.40,        # Defensive
        'risk_off_sentiment': +0.35,    # Flight to safety
        'inflation_pressure': -0.30,    # Rate sensitive
        'regulation_risk': -0.25,
    },
    'Real Estate': {
        'inflation_pressure': -0.50,    # Rate sensitive
        'recession_risk': -0.40,
        'credit_stress': -0.40,
        'risk_off_sentiment': -0.20,
    },
    'Communication Services': {
        'recession_risk': -0.30,        # Ad spending cyclical
        'regulation_risk': -0.35,       # Content, antitrust
        'trade_war_risk': -0.20,
        'risk_off_sentiment': -0.25,
    },
    # Asset classes
    'Bonds_Long': {
        'inflation_pressure': -0.80,    # Duration risk
        'recession_risk': +0.60,        # Flight to safety
        'risk_off_sentiment': +0.70,
        'fx_stress': +0.20,             # USD strength
    },
    'Bonds_Short': {
        'inflation_pressure': -0.20,    # Less duration
        'recession_risk': +0.30,
        'risk_off_sentiment': +0.30,
    },
    'Gold': {
        'inflation_pressure': +0.50,    # Inflation hedge
        'geopolitical_tension': +0.60,  # Safe haven
        'risk_off_sentiment': +0.70,
        'fx_stress': +0.40,             # Dollar hedge
        'conflict_risk': +0.50,
    },
    'Oil': {
        'oil_supply_shock': +0.90,
        'conflict_risk': +0.50,
        'sanctions_risk': +0.40,
        'recession_risk': -0.60,        # Demand destruction
    },
}

# Ticker to sector mapping (extend as needed)
TICKER_SECTOR_MAP = {
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'OXY': 'Energy', 'EOG': 'Energy', 'XLE': 'Energy',

    # Financials
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
    'GS': 'Financials', 'MS': 'Financials', 'C': 'Financials',
    'XLF': 'Financials',

    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
    'NVDA': 'Technology', 'META': 'Technology', 'AMZN': 'Technology',
    'AMD': 'Technology', 'INTC': 'Technology', 'XLK': 'Technology',

    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
    'MRK': 'Healthcare', 'ABBV': 'Healthcare', 'XLV': 'Healthcare',

    # Consumer Discretionary
    'TSLA': 'Consumer Discretionary', 'HD': 'Consumer Discretionary',
    'NKE': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
    'XLY': 'Consumer Discretionary',

    # Consumer Staples
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
    'WMT': 'Consumer Staples', 'COST': 'Consumer Staples', 'XLP': 'Consumer Staples',

    # Industrials
    'CAT': 'Industrials', 'BA': 'Industrials', 'HON': 'Industrials',
    'UPS': 'Industrials', 'LMT': 'Industrials', 'RTX': 'Industrials',
    'XLI': 'Industrials',

    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'NEM': 'Materials',
    'FCX': 'Materials', 'XLB': 'Materials',

    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities',
    'XLU': 'Utilities',

    # Real Estate
    'AMT': 'Real Estate', 'PLD': 'Real Estate', 'XLRE': 'Real Estate',

    # Communication Services
    'DIS': 'Communication Services', 'NFLX': 'Communication Services',
    'T': 'Communication Services', 'VZ': 'Communication Services',
    'XLC': 'Communication Services',

    # Bonds
    'TLT': 'Bonds_Long', 'ZROZ': 'Bonds_Long', 'EDV': 'Bonds_Long',
    'TMF': 'Bonds_Long', 'IEF': 'Bonds_Long',
    'SHY': 'Bonds_Short', 'BIL': 'Bonds_Short',
    'TBT': 'Bonds_Long',  # Inverse, but same sensitivity (reversed in calculation)

    # Commodities
    'GLD': 'Gold', 'IAU': 'Gold', 'GDX': 'Gold',
    'USO': 'Oil', 'XOP': 'Oil',
}


# =============================================================================
# EVENT CLASSIFICATION RULES
# =============================================================================

# Keyword patterns for event classification (rule-based first pass)
EVENT_CLASSIFICATION_RULES = {
    EventType.CONFLICT: {
        'keywords': ['war', 'military', 'invasion', 'attack', 'strike', 'troops',
                     'missile', 'bombing', 'combat', 'armed forces', 'defense'],
        'entities': ['ukraine', 'russia', 'israel', 'gaza', 'iran', 'china', 'taiwan'],
        'factor_impacts': {'conflict_risk': 20, 'geopolitical_tension': 15, 'oil_supply_shock': 10},
    },
    EventType.SANCTIONS: {
        'keywords': ['sanction', 'embargo', 'asset freeze', 'trade restriction',
                     'blacklist', 'export ban', 'import ban'],
        'entities': ['russia', 'china', 'iran', 'north korea', 'venezuela'],
        'factor_impacts': {'sanctions_risk': 25, 'trade_war_risk': 10, 'supply_chain_stress': 10},
    },
    EventType.COUP_POLITICAL_CRISIS: {
        'keywords': ['coup', 'overthrow', 'regime change', 'political crisis',
                     'impeachment', 'resignation', 'arrested president'],
        'factor_impacts': {'political_uncertainty': 25, 'geopolitical_tension': 15},
    },
    EventType.TARIFF: {
        'keywords': ['tariff', 'trade war', 'import duty', 'export duty', 'trade barrier',
                     'trade deal', 'trade agreement', 'trade talks'],
        'entities': ['china', 'eu', 'mexico', 'canada'],
        'factor_impacts': {'trade_war_risk': 20, 'inflation_pressure': 5},
    },
    EventType.OPEC_DECISION: {
        'keywords': ['opec', 'oil production', 'output cut', 'production quota',
                     'barrel', 'crude oil', 'petroleum'],
        'entities': ['opec', 'saudi arabia', 'russia', 'uae'],
        'factor_impacts': {'oil_supply_shock': 25, 'energy_disruption': 15, 'inflation_pressure': 10},
    },
    EventType.ENERGY_DISRUPTION: {
        'keywords': ['pipeline', 'refinery', 'oil spill', 'outage', 'disruption',
                     'shutdown', 'explosion', 'natural gas'],
        'factor_impacts': {'energy_disruption': 20, 'oil_supply_shock': 15, 'supply_chain_stress': 10},
    },
    EventType.CENTRAL_BANK: {
        'keywords': ['federal reserve', 'fed', 'ecb', 'boj', 'bank of england',
                     'rate hike', 'rate cut', 'monetary policy', 'quantitative',
                     'powell', 'lagarde'],
        'factor_impacts': {'inflation_pressure': 10},  # Direction depends on content
    },
    EventType.INFLATION_DATA: {
        'keywords': ['cpi', 'inflation', 'consumer price', 'pce', 'core inflation',
                     'price index', 'cost of living'],
        'factor_impacts': {'inflation_pressure': 15},
    },
    EventType.RECESSION_SIGNAL: {
        'keywords': ['recession', 'gdp', 'economic contraction', 'unemployment',
                     'jobless', 'layoffs', 'downturn', 'economic slowdown'],
        'factor_impacts': {'recession_risk': 20, 'risk_off_sentiment': 10},
    },
    EventType.FX_CRISIS: {
        'keywords': ['currency crisis', 'devaluation', 'forex', 'exchange rate',
                     'peso', 'lira', 'yuan', 'emerging market'],
        'factor_impacts': {'fx_stress': 20, 'risk_off_sentiment': 10},
    },
    EventType.ELECTION: {
        'keywords': ['election', 'vote', 'poll', 'ballot', 'president-elect',
                     'inauguration', 'campaign'],
        'factor_impacts': {'political_uncertainty': 15},
    },
    EventType.REGULATION: {
        'keywords': ['regulation', 'antitrust', 'regulatory', 'sec', 'ftc',
                     'legislation', 'bill passed', 'law enacted'],
        'factor_impacts': {'regulation_risk': 15},
    },
}

# Entity extraction patterns
ENTITY_PATTERNS = {
    'countries': [
        'united states', 'usa', 'us', 'china', 'russia', 'ukraine', 'iran',
        'saudi arabia', 'venezuela', 'mexico', 'canada', 'germany', 'france',
        'uk', 'britain', 'japan', 'india', 'brazil', 'israel', 'taiwan',
        'north korea', 'south korea', 'turkey', 'egypt', 'iraq', 'syria',
    ],
    'commodities': [
        'oil', 'crude', 'natural gas', 'gold', 'silver', 'copper', 'wheat',
        'corn', 'soybeans', 'lithium', 'uranium', 'coal',
    ],
    'organizations': [
        'opec', 'fed', 'federal reserve', 'ecb', 'imf', 'world bank',
        'nato', 'un', 'united nations', 'eu', 'european union', 'wto',
    ],
}


# =============================================================================
# MACRO EVENT ENGINE
# =============================================================================

class MacroEventEngine:
    """
    Macro/Geopolitical Event Engine.

    Ingests news, deduplicates into events, computes deterministic factor scores,
    and maps to sector/asset exposures.
    """

    def __init__(self):
        self.events: Dict[str, MacroEvent] = {}  # event_id -> MacroEvent
        self.factors = MacroFactorScores()
        self._last_update: Optional[datetime] = None
        self._headline_cache: Set[str] = set()  # For deduplication

        logger.info("MacroEventEngine initialized")

    # =========================================================================
    # EVENT INGESTION
    # =========================================================================

    def ingest_headlines(self, headlines: List[Dict]) -> List[MacroEvent]:
        """
        Ingest headlines and convert to deduplicated events.

        Args:
            headlines: List of dicts with keys: title, source, published_at, url, description

        Returns:
            List of new/updated MacroEvents
        """
        new_events = []

        for headline in headlines:
            title = headline.get('title') or headline.get('headline', '')
            if not title:
                continue

            # Skip if we've seen this exact headline
            title_hash = hashlib.md5(title.lower().encode()).hexdigest()[:12]
            if title_hash in self._headline_cache:
                continue
            self._headline_cache.add(title_hash)

            # Classify the headline
            event_type, severity, entities, factor_impacts = self._classify_headline(title, headline.get('description', ''))

            if event_type == EventType.UNKNOWN:
                continue  # Skip non-macro headlines

            # Generate event ID (for deduplication across similar headlines)
            event_id = self._generate_event_id(event_type, entities, title)

            # Get or create event
            if event_id in self.events:
                # Update existing event
                event = self.events[event_id]
                event.headline_count += 1
                event.last_updated = datetime.now()

                # Add source if new
                source = headline.get('source', 'unknown').lower()
                if source not in event.sources:
                    event.sources.append(source)
                    event.source_count += 1

                # Recalculate confidence
                event.confidence = self._calculate_event_confidence(event)
            else:
                # Create new event
                source = headline.get('source', 'unknown').lower()
                published = headline.get('published_at')
                if isinstance(published, str):
                    try:
                        published = datetime.fromisoformat(published.replace('Z', '+00:00'))
                    except:
                        published = datetime.now()
                elif not published:
                    published = datetime.now()

                event = MacroEvent(
                    event_id=event_id,
                    event_type=event_type,
                    severity=severity,
                    title=self._generate_event_title(event_type, entities, title),
                    summary=headline.get('description', title)[:200],
                    entities=entities,
                    first_seen=published if isinstance(published, datetime) else datetime.now(),
                    last_updated=datetime.now(),
                    sources=[source],
                    source_count=1,
                    headline_count=1,
                    factor_impacts=factor_impacts,
                )
                event.confidence = self._calculate_event_confidence(event)
                self.events[event_id] = event
                new_events.append(event)

        return new_events

    def _classify_headline(self, title: str, description: str = '') -> Tuple[EventType, EventSeverity, Dict, Dict]:
        """
        Classify headline using rule-based approach.
        Returns: (event_type, severity, entities, factor_impacts)
        """
        text = (title + ' ' + description).lower()

        best_match = None
        best_score = 0

        for event_type, rules in EVENT_CLASSIFICATION_RULES.items():
            score = 0

            # Keyword matching
            for keyword in rules.get('keywords', []):
                if keyword in text:
                    score += 2

            # Entity matching (bonus)
            for entity in rules.get('entities', []):
                if entity in text:
                    score += 1

            if score > best_score:
                best_score = score
                best_match = event_type

        if best_score < 2:  # Minimum threshold
            return EventType.UNKNOWN, EventSeverity.MINIMAL, {}, {}

        # Extract entities
        entities = self._extract_entities(text)

        # Get factor impacts from rules
        factor_impacts = EVENT_CLASSIFICATION_RULES.get(best_match, {}).get('factor_impacts', {}).copy()

        # Determine severity based on keywords and source
        severity = self._determine_severity(text, best_score)

        # Adjust factor impacts based on severity
        severity_multiplier = {
            EventSeverity.CRITICAL: 1.5,
            EventSeverity.HIGH: 1.2,
            EventSeverity.MEDIUM: 1.0,
            EventSeverity.LOW: 0.7,
            EventSeverity.MINIMAL: 0.4,
        }
        for factor in factor_impacts:
            factor_impacts[factor] = int(factor_impacts[factor] * severity_multiplier.get(severity, 1.0))

        return best_match, severity, entities, factor_impacts

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        entities = {
            'countries': [],
            'commodities': [],
            'organizations': [],
        }

        text_lower = text.lower()

        for entity_type, patterns in ENTITY_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    entities[entity_type].append(pattern)

        return entities

    def _determine_severity(self, text: str, keyword_score: int) -> EventSeverity:
        """Determine event severity."""
        critical_words = ['war', 'invasion', 'crisis', 'crash', 'collapse', 'emergency']
        high_words = ['attack', 'sanction', 'tariff', 'recession', 'surge', 'plunge']

        for word in critical_words:
            if word in text:
                return EventSeverity.CRITICAL

        for word in high_words:
            if word in text:
                return EventSeverity.HIGH

        if keyword_score >= 5:
            return EventSeverity.HIGH
        elif keyword_score >= 3:
            return EventSeverity.MEDIUM
        else:
            return EventSeverity.LOW

    def _generate_event_id(self, event_type: EventType, entities: Dict, title: str) -> str:
        """Generate a stable event ID for deduplication."""
        # Combine event type + key entities
        key_parts = [event_type.value]

        # Add top entities
        for country in entities.get('countries', [])[:2]:
            key_parts.append(country)
        for commodity in entities.get('commodities', [])[:1]:
            key_parts.append(commodity)

        # If no entities, use title hash
        if len(key_parts) == 1:
            key_parts.append(hashlib.md5(title.lower().encode()).hexdigest()[:8])

        return '_'.join(key_parts)

    def _generate_event_title(self, event_type: EventType, entities: Dict, original_title: str) -> str:
        """Generate canonical event title."""
        countries = entities.get('countries', [])

        if event_type == EventType.CONFLICT and countries:
            return f"{countries[0].title()} Conflict"
        elif event_type == EventType.SANCTIONS and countries:
            return f"Sanctions on {countries[0].title()}"
        elif event_type == EventType.OPEC_DECISION:
            return "OPEC Production Decision"
        elif event_type == EventType.TARIFF and countries:
            return f"Trade Tensions with {countries[0].title()}"
        else:
            # Use original title, truncated
            return original_title[:60]

    def _calculate_event_confidence(self, event: MacroEvent) -> int:
        """
        Calculate event confidence based on sources.
        Deterministic formula: source_tier_weights * cross_source_bonus * recency
        """
        if event.source_count == 0:
            return 30

        # Base confidence from source tiers
        total_weight = 0
        for source in event.sources:
            tier_weight = SOURCE_TIERS.get(source, 0.3)
            total_weight += tier_weight

        avg_tier = total_weight / event.source_count
        base_confidence = int(avg_tier * 60)  # Max 60 from tier

        # Cross-source confirmation bonus (up to +25)
        cross_source_bonus = min(25, event.source_count * 5)

        # Recency factor
        age_hours = (datetime.now() - event.first_seen).total_seconds() / 3600
        if age_hours <= RECENCY_DECAY['full_weight_hours']:
            recency_factor = 1.0
        else:
            half_life = RECENCY_DECAY['half_life_hours']
            recency_factor = max(
                RECENCY_DECAY['min_weight'],
                0.5 ** ((age_hours - RECENCY_DECAY['full_weight_hours']) / half_life)
            )

        # Headline count bonus (up to +15)
        headline_bonus = min(15, event.headline_count * 2)

        confidence = int((base_confidence + cross_source_bonus + headline_bonus) * recency_factor)
        return min(100, max(10, confidence))

    # =========================================================================
    # FACTOR SCORE COMPUTATION
    # =========================================================================

    def compute_factor_scores(self) -> MacroFactorScores:
        """
        Compute macro factor scores from active events.
        DETERMINISTIC: weighted sum of event impacts with decay.
        """
        factors = MacroFactorScores(timestamp=datetime.now())

        # Aggregate factor deltas from all active events
        factor_deltas = defaultdict(float)
        active_events = []

        now = datetime.now()
        stale_threshold = timedelta(hours=RECENCY_DECAY['stale_hours'])

        for event in self.events.values():
            # Skip stale events
            age = now - event.last_updated
            if age > stale_threshold:
                event.is_ongoing = False
                continue

            active_events.append(event)

            # Calculate recency weight
            age_hours = age.total_seconds() / 3600
            if age_hours <= RECENCY_DECAY['full_weight_hours']:
                recency_weight = 1.0
            else:
                half_life = RECENCY_DECAY['half_life_hours']
                recency_weight = max(
                    RECENCY_DECAY['min_weight'],
                    0.5 ** ((age_hours - RECENCY_DECAY['full_weight_hours']) / half_life)
                )

            # Apply event's factor impacts
            confidence_weight = event.confidence / 100.0

            for factor, delta in event.factor_impacts.items():
                weighted_delta = delta * recency_weight * confidence_weight
                factor_deltas[factor] += weighted_delta

        # Apply deltas to baseline (50)
        factor_mapping = {
            'oil_supply_shock': 'oil_supply_shock',
            'sanctions_risk': 'sanctions_risk',
            'conflict_risk': 'conflict_risk',
            'geopolitical_tension': 'geopolitical_tension',
            'trade_war_risk': 'trade_war_risk',
            'regulation_risk': 'regulation_risk',
            'fiscal_stimulus': 'fiscal_stimulus',
            'political_uncertainty': 'political_uncertainty',
            'inflation_pressure': 'inflation_pressure',
            'recession_risk': 'recession_risk',
            'risk_off_sentiment': 'risk_off_sentiment',
            'fx_stress': 'fx_stress',
            'credit_stress': 'credit_stress',
            'energy_disruption': 'energy_disruption',
            'supply_chain_stress': 'supply_chain_stress',
            'commodity_volatility': 'commodity_volatility',
        }

        for factor_key, attr_name in factor_mapping.items():
            delta = factor_deltas.get(factor_key, 0)
            new_value = 50 + delta
            new_value = max(0, min(100, int(new_value)))
            setattr(factors, attr_name, new_value)

        # Metadata
        factors.active_event_count = len(active_events)
        factors.data_freshness = "FRESH" if active_events else "NO_DATA"

        if active_events:
            factors.last_event_time = max(e.last_updated for e in active_events)

        self.factors = factors
        self._last_update = now

        return factors

    # =========================================================================
    # EXPOSURE MAPPING
    # =========================================================================

    def compute_portfolio_exposure(
        self,
        portfolio_tickers: List[str],
        ticker_sectors: Optional[Dict[str, str]] = None
    ) -> PortfolioMacroExposure:
        """
        Compute how macro factors impact the portfolio.

        Args:
            portfolio_tickers: List of tickers in portfolio
            ticker_sectors: Optional mapping of ticker -> sector
        """
        if ticker_sectors is None:
            ticker_sectors = {}

        # Ensure we have current factor scores
        if self._last_update is None or (datetime.now() - self._last_update).seconds > 300:
            self.compute_factor_scores()

        factors = self.factors
        sector_exposures = {}
        ticker_impacts = {}

        # Compute sector exposures
        unique_sectors = set()
        for ticker in portfolio_tickers:
            sector = ticker_sectors.get(ticker) or TICKER_SECTOR_MAP.get(ticker, 'Unknown')
            unique_sectors.add(sector)

        for sector in unique_sectors:
            sensitivities = SECTOR_FACTOR_SENSITIVITY.get(sector, {})

            # Calculate net impact
            net_impact = 0.0
            tailwinds = []
            headwinds = []

            for factor_name, sensitivity in sensitivities.items():
                factor_value = getattr(factors, factor_name, 50)
                factor_deviation = (factor_value - 50) / 50  # -1 to +1

                impact = factor_deviation * sensitivity * 10  # Scale to reasonable range
                net_impact += impact

                if impact > 1:
                    tailwinds.append(f"{factor_name}: +{impact:.1f}")
                elif impact < -1:
                    headwinds.append(f"{factor_name}: {impact:.1f}")

            sector_exposures[sector] = SectorExposure(
                sector=sector,
                factor_sensitivities=sensitivities,
                current_impact=net_impact,
                tailwinds=tailwinds,
                headwinds=headwinds,
            )

        # Compute per-ticker impacts
        for ticker in portfolio_tickers:
            sector = ticker_sectors.get(ticker) or TICKER_SECTOR_MAP.get(ticker, 'Unknown')

            if sector in sector_exposures:
                base_impact = sector_exposures[sector].current_impact

                # Special handling for inverse ETFs
                if ticker in ['TBT', 'SQQQ', 'SH']:
                    base_impact = -base_impact

                ticker_impacts[ticker] = base_impact
            else:
                ticker_impacts[ticker] = 0.0

        # Aggregate portfolio impact
        if ticker_impacts:
            net_portfolio_impact = sum(ticker_impacts.values()) / len(ticker_impacts)
        else:
            net_portfolio_impact = 0.0

        # Top tailwinds/headwinds
        sorted_impacts = sorted(ticker_impacts.items(), key=lambda x: x[1], reverse=True)
        top_tailwinds = [(t, i) for t, i in sorted_impacts if i > 0][:5]
        top_headwinds = [(t, i) for t, i in sorted_impacts if i < 0][-5:]

        # Generate alerts
        alerts = []
        elevated = factors.get_elevated_factors(threshold=65)
        for factor, value in elevated[:3]:
            alerts.append(f"⚠️ {factor.replace('_', ' ').title()} elevated: {value}")

        return PortfolioMacroExposure(
            timestamp=datetime.now(),
            factors=factors,
            sector_exposures=sector_exposures,
            ticker_impacts=ticker_impacts,
            net_portfolio_impact=net_portfolio_impact,
            top_tailwinds=top_tailwinds,
            top_headwinds=top_headwinds,
            alerts=alerts,
        )

    # =========================================================================
    # DATA FETCHING (using existing sources)
    # =========================================================================

    def fetch_macro_news(self) -> List[Dict]:
        """
        Fetch macro/geopolitical news using existing platform sources.
        Uses: news DB, web search tools.
        """
        headlines = []

        # Source 1: Database - recent non-ticker headlines
        try:
            headlines.extend(self._fetch_from_database())
        except Exception as e:
            logger.warning(f"Database fetch failed: {e}")

        # Source 2: Google News RSS for macro topics
        try:
            headlines.extend(self._fetch_google_news_macro())
        except Exception as e:
            logger.warning(f"Google News fetch failed: {e}")

        return headlines

    def _fetch_from_database(self) -> List[Dict]:
        """Fetch recent headlines from news_articles table."""
        headlines = []

        try:
            with get_connection() as conn:
                # Get recent macro-relevant headlines
                query = """
                    SELECT DISTINCT title, source, published_at, url, summary as description
                    FROM news_articles
                    WHERE published_at > NOW() - INTERVAL '48 hours'
                    AND (
                        title ILIKE ANY(ARRAY[
                            '%war%', '%sanction%', '%tariff%', '%opec%', '%inflation%',
                            '%recession%', '%fed%', '%central bank%', '%crisis%',
                            '%election%', '%president%', '%oil%', '%geopolitical%'
                        ])
                        OR ticker IS NULL
                    )
                    ORDER BY published_at DESC
                    LIMIT 100
                """
                result = conn.execute(query)

                for row in result:
                    headlines.append({
                        'title': row.title,
                        'source': row.source,
                        'published_at': row.published_at,
                        'url': row.url,
                        'description': row.description,
                    })
        except Exception as e:
            logger.debug(f"Database query failed: {e}")

        return headlines

    def _fetch_google_news_macro(self) -> List[Dict]:
        """Fetch macro news from Google News RSS."""
        import feedparser

        headlines = []

        # Macro-focused search queries
        macro_queries = [
            'geopolitical+news',
            'oil+price+opec',
            'trade+war+tariff',
            'federal+reserve+rate',
            'global+economy+recession',
        ]

        for query in macro_queries[:2]:  # Limit to avoid rate limits
            try:
                url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(url)

                for entry in feed.entries[:10]:
                    # Extract source from title (Google News format: "Title - Source")
                    title = entry.get('title', '')
                    source = 'google news'
                    if ' - ' in title:
                        parts = title.rsplit(' - ', 1)
                        title = parts[0]
                        source = parts[1].lower() if len(parts) > 1 else 'google news'

                    headlines.append({
                        'title': title,
                        'source': source,
                        'published_at': entry.get('published', ''),
                        'url': entry.get('link', ''),
                        'description': entry.get('summary', ''),
                    })
            except Exception as e:
                logger.debug(f"Google News fetch failed for {query}: {e}")

        return headlines

    # =========================================================================
    # OUTPUT GENERATION
    # =========================================================================

    def get_macro_context_for_ai(self, portfolio_tickers: Optional[List[str]] = None) -> str:
        """
        Generate macro context block for AI prompts.
        READ-ONLY: provides context, doesn't override decisions.
        """
        self.compute_factor_scores()

        context_parts = []
        context_parts.append("=" * 50)
        context_parts.append("MACRO/GEOPOLITICAL CONTEXT")
        context_parts.append("=" * 50)

        # Active events
        active_events = [e for e in self.events.values() if e.is_ongoing]
        if active_events:
            context_parts.append(f"\n📰 ACTIVE EVENTS ({len(active_events)}):")
            for event in sorted(active_events, key=lambda e: e.confidence, reverse=True)[:5]:
                context_parts.append(
                    f"  • {event.title} [{event.event_type.value}] "
                    f"(confidence: {event.confidence}%, sources: {event.source_count})"
                )
        else:
            context_parts.append("\n📰 No significant macro events detected")

        # Factor scores
        context_parts.append(f"\n📊 MACRO FACTOR SCORES (50=neutral):")
        elevated = self.factors.get_elevated_factors(threshold=55)
        reduced = self.factors.get_reduced_factors(threshold=45)

        if elevated:
            context_parts.append("  Elevated risks:")
            for factor, value in elevated[:5]:
                context_parts.append(f"    • {factor.replace('_', ' ')}: {value}")

        if reduced:
            context_parts.append("  Reduced risks:")
            for factor, value in reduced:
                context_parts.append(f"    • {factor.replace('_', ' ')}: {value}")

        # Portfolio exposure if provided
        if portfolio_tickers:
            exposure = self.compute_portfolio_exposure(portfolio_tickers)

            context_parts.append(f"\n💼 PORTFOLIO MACRO EXPOSURE:")
            context_parts.append(f"  Net impact: {exposure.net_portfolio_impact:+.1f}")

            if exposure.top_tailwinds:
                context_parts.append("  Tailwinds:")
                for ticker, impact in exposure.top_tailwinds[:3]:
                    context_parts.append(f"    • {ticker}: +{impact:.1f}")

            if exposure.top_headwinds:
                context_parts.append("  Headwinds:")
                for ticker, impact in exposure.top_headwinds[:3]:
                    context_parts.append(f"    • {ticker}: {impact:.1f}")

            if exposure.alerts:
                context_parts.append("  Alerts:")
                for alert in exposure.alerts:
                    context_parts.append(f"    {alert}")

        context_parts.append("=" * 50)

        return "\n".join(context_parts)

    def get_active_events(self) -> List[MacroEvent]:
        """Get list of active macro events."""
        return [e for e in self.events.values() if e.is_ongoing]

    def _deduplicate_events(self) -> int:
        """
        Deduplicate macro events by event type and date window.
        Events of the same type within 3 days are considered duplicates.
        Keep the one with highest confidence.

        Returns:
            Number of duplicates removed
        """
        if not self.events:
            return 0

        original_count = len(self.events)

        # Group by event type
        type_groups: Dict[str, List[MacroEvent]] = defaultdict(list)
        for event in self.events.values():
            key = event.event_type.value
            type_groups[key].append(event)

        # Keep only best event per 3-day window within each type
        deduped_events: Dict[str, MacroEvent] = {}

        for event_type, group in type_groups.items():
            # Sort by last_updated
            group.sort(key=lambda e: e.last_updated)

            # Keep only one event per 3-day window, preferring highest confidence
            kept: List[MacroEvent] = []
            for event in group:
                is_duplicate = False
                for i, kept_event in enumerate(kept):
                    # Check if within 3-day window
                    time_diff = abs((event.last_updated - kept_event.last_updated).days)
                    if time_diff <= 3:
                        # Same type, within 3 days - it's a duplicate
                        # Keep the one with higher confidence
                        if event.confidence > kept_event.confidence:
                            kept[i] = event  # Replace with higher confidence
                        is_duplicate = True
                        break

                if not is_duplicate:
                    kept.append(event)

            # Add kept events to deduped dict
            for event in kept:
                deduped_events[event.event_id] = event

        # Update events dict
        removed_count = original_count - len(deduped_events)
        self.events = deduped_events

        if removed_count > 0:
            logger.info(f"Deduplicated events: {original_count} -> {len(deduped_events)} "
                       f"({removed_count} duplicates removed)")

        return removed_count

    def refresh(self) -> Tuple[int, MacroFactorScores]:
        """
        Full refresh: fetch news, ingest, deduplicate, compute factors.
        Returns: (new_event_count, updated_factors)
        """
        headlines = self.fetch_macro_news()
        new_events = self.ingest_headlines(headlines)

        # FIX: Deduplicate events before computing factor scores
        # This prevents the same event (e.g., "OPEC decision") from appearing
        # multiple times from different news sources and inflating factor scores
        duplicates_removed = self._deduplicate_events()

        factors = self.compute_factor_scores()

        logger.info(f"MacroEventEngine refreshed: {len(new_events)} new events, "
                   f"{duplicates_removed} duplicates removed, "
                   f"{factors.active_event_count} active events")

        return len(new_events), factors


# =============================================================================
# SINGLETON AND CONVENIENCE FUNCTIONS
# =============================================================================

_engine: Optional[MacroEventEngine] = None


def get_macro_engine() -> MacroEventEngine:
    """Get or create the macro event engine singleton."""
    global _engine
    if _engine is None:
        _engine = MacroEventEngine()
    return _engine


def get_macro_factors() -> MacroFactorScores:
    """Get current macro factor scores."""
    engine = get_macro_engine()
    return engine.compute_factor_scores()


def get_macro_context(portfolio_tickers: Optional[List[str]] = None) -> str:
    """Get macro context for AI prompts."""
    engine = get_macro_engine()
    return engine.get_macro_context_for_ai(portfolio_tickers)


def refresh_macro_data() -> Tuple[int, MacroFactorScores]:
    """Refresh macro data from all sources."""
    engine = get_macro_engine()
    return engine.refresh()


# =============================================================================
# MAIN - TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MACRO EVENT ENGINE - TEST")
    print("=" * 70)

    engine = MacroEventEngine()

    # Test with sample headlines
    test_headlines = [
        {
            'title': 'US captures Venezuelan president Maduro in military operation',
            'source': 'Reuters',
            'published_at': datetime.now(),
            'description': 'US forces captured Venezuelan President Maduro. Oil investors eye opportunities.',
        },
        {
            'title': 'OPEC+ agrees to cut oil production by 2 million barrels per day',
            'source': 'Bloomberg',
            'published_at': datetime.now() - timedelta(hours=2),
            'description': 'OPEC and allies agree to significant production cuts to boost prices.',
        },
        {
            'title': 'China tariffs on US goods to increase by 25%',
            'source': 'Wall Street Journal',
            'published_at': datetime.now() - timedelta(hours=5),
            'description': 'Trade tensions escalate as China announces retaliatory tariffs.',
        },
        {
            'title': 'Federal Reserve signals pause in rate hikes',
            'source': 'CNBC',
            'published_at': datetime.now() - timedelta(hours=1),
            'description': 'Fed Chair Powell indicates rates may stay steady at next meeting.',
        },
        {
            'title': 'Russia-Ukraine conflict intensifies near border',
            'source': 'BBC',
            'published_at': datetime.now() - timedelta(hours=3),
            'description': 'Military activity increases along the Russia-Ukraine border.',
        },
    ]

    print("\n📰 INGESTING TEST HEADLINES...")
    new_events = engine.ingest_headlines(test_headlines)
    print(f"   Created {len(new_events)} events")

    for event in engine.get_active_events():
        print(f"   • {event.title} [{event.event_type.value}] - confidence: {event.confidence}%")

    print("\n📊 COMPUTING FACTOR SCORES...")
    factors = engine.compute_factor_scores()

    print(f"\n   Factor Scores (50 = neutral):")
    for factor, value in factors.get_elevated_factors(threshold=50):
        emoji = "🔴" if value >= 70 else "🟡" if value >= 60 else "🟢"
        print(f"   {emoji} {factor}: {value}")

    print("\n💼 PORTFOLIO EXPOSURE...")
    test_portfolio = ['AAPL', 'XOM', 'TLT', 'JPM', 'CVX', 'DAL', 'GLD']
    exposure = engine.compute_portfolio_exposure(test_portfolio)

    print(f"   Net Portfolio Impact: {exposure.net_portfolio_impact:+.1f}")
    print(f"   Tailwinds: {exposure.top_tailwinds}")
    print(f"   Headwinds: {exposure.top_headwinds}")

    print("\n🤖 AI CONTEXT BLOCK:")
    print(engine.get_macro_context_for_ai(test_portfolio))

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)