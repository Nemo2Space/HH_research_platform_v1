"""
HH Research Platform - Sukuk Domain Models

Defines data structures for USD Sukuk trading via IBKR.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, List, Dict, Any
from enum import Enum


# Default issuer mapping for HYSK holdings
DEFAULT_ISSUER_MAPPING = {
    "PIFKSA": {"issuer": "Public Investment Fund (PIF)", "bucket": "KSA_SOVEREIGN_WEALTH"},
    "SHARSK": {"issuer": "Sharjah Sukuk Programme", "bucket": "UAE_SOVEREIGN"},
    "ENEDEV": {"issuer": "Energy Development Oman (EDO)", "bucket": "OMAN_CORPORATE"},
    "AL": {"issuer": "Air Lease Corporation", "bucket": "US_CORPORATE"},
    "ESICSU": {"issuer": "Emirates Steel Industries", "bucket": "UAE_CORPORATE"},
    "GASBCM": {"issuer": "SOCAR (TMS Issuer)", "bucket": "AZERBAIJAN_CORPORATE"},
    "ADIBUH": {"issuer": "Abu Dhabi Islamic Bank", "bucket": "UAE_BANK"},
    "SECO": {"issuer": "Saudi Electricity Company", "bucket": "KSA_CORPORATE"},
    "INDOIS": {"issuer": "Republic of Indonesia", "bucket": "INDONESIA_SOVEREIGN"},
    "QIBKQD": {"issuer": "Qatar Islamic Bank", "bucket": "QATAR_BANK"},
    "DPWDU": {"issuer": "DP World", "bucket": "UAE_CORPORATE"},
    "ALDAR": {"issuer": "Aldar Properties", "bucket": "UAE_CORPORATE"},
    "DIBUH": {"issuer": "Dubai Islamic Bank", "bucket": "UAE_BANK"},
    "KFHKK": {"issuer": "Kuwait Finance House", "bucket": "KUWAIT_BANK"},
    "SRCSUK": {"issuer": "Saudi Real Estate Refinance Company", "bucket": "KSA_CORPORATE"},
    # Add more as needed
}


class DataQuality(Enum):
    """Data quality levels for risk management."""
    OK = "OK"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    MISSING = "MISSING"
    SUSPICIOUS = "SUSPICIOUS"


class SukukAction(Enum):
    """Trading signal actions."""
    BUY = "BUY"
    HOLD = "HOLD"
    WATCH = "WATCH"
    AVOID = "AVOID"


class MaturityBucket(Enum):
    """Time-to-maturity buckets for ladder strategy."""
    SHORT = "SHORT"      # 0-3 years
    MEDIUM = "MEDIUM"    # 3-7 years
    LONG = "LONG"        # 7-15 years
    ULTRA = "ULTRA"      # 15+ years


@dataclass
class SukukInstrument:
    """
    Sukuk instrument definition from universe JSON.

    Represents a single sukuk bond available for trading on IBKR.
    """
    conid: int
    isin: str
    name: str
    issuer: str
    issuer_bucket: str  # e.g., "KSA_SOVEREIGN", "UAE_BANK"
    maturity: date
    coupon_rate_pct: float
    face_value: float
    min_size: float
    currency: str
    exchange: str
    weight: float  # Target portfolio weight (0-1)
    original_name: str = ""
    coupon_frequency: int = 12  # payments per year (12 = monthly, 2 = semi-annual)
    callable: bool = False
    ticker_prefix: str = ""  # e.g., "PIFKSA", "SECO" - used for issuer lookup
    cached_price: Optional[float] = None  # Last known price from JSON
    cached_yield: Optional[float] = None  # Last known askYield from JSON
    cached_ttm: Optional[float] = None    # Last known TTM from JSON
    notes: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any], issuer_mapping: Dict[str, Dict] = None) -> 'SukukInstrument':
        """
        Create from JSON dictionary.

        Args:
            d: Dictionary from JSON
            issuer_mapping: Optional dict mapping ticker prefix to {issuer, bucket}
        """
        # Parse maturity - handle both YYYY-MM-DD and MM/DD/YY formats
        maturity_str = d.get('maturity', '')
        if isinstance(maturity_str, str) and maturity_str:
            try:
                if '-' in maturity_str:
                    maturity = datetime.strptime(maturity_str, '%Y-%m-%d').date()
                elif '/' in maturity_str:
                    maturity = datetime.strptime(maturity_str, '%m/%d/%y').date()
                else:
                    maturity = date.today()
            except ValueError:
                maturity = date.today()
        else:
            maturity = date.today()

        # Get ticker prefix (country field in HYSK JSON)
        ticker_prefix = d.get('country', d.get('ticker_prefix', ''))

        # Look up issuer info from mapping
        issuer = d.get('issuer', '')
        issuer_bucket = d.get('issuer_bucket', '')

        if issuer_mapping and ticker_prefix in issuer_mapping:
            mapping = issuer_mapping[ticker_prefix]
            issuer = issuer or mapping.get('issuer', ticker_prefix)
            issuer_bucket = issuer_bucket or mapping.get('bucket', 'OTHER')
        elif not issuer:
            # Use original name or ticker as fallback
            issuer = d.get('originalName', ticker_prefix) or ticker_prefix

        if not issuer_bucket:
            issuer_bucket = ticker_prefix or 'OTHER'

        return cls(
            conid=d.get('conid', 0),
            isin=d.get('isin', ''),
            name=d.get('name', d.get('symbol', '')),
            issuer=issuer,
            issuer_bucket=issuer_bucket,
            maturity=maturity,
            coupon_rate_pct=float(d.get('coupon_rate_pct', d.get('coupon', 0))),
            face_value=d.get('face_value', 1000),
            min_size=d.get('min_size', d.get('minSize', 200)),
            currency=d.get('currency', 'USD'),
            exchange=d.get('exchange', 'SMART'),
            weight=d.get('weight', 0.0),
            original_name=d.get('original_name', d.get('originalName', '')),
            coupon_frequency=d.get('coupon_frequency', 12),
            callable=d.get('callable', False),
            ticker_prefix=ticker_prefix,
            cached_price=d.get('last_price', d.get('price')),
            cached_yield=d.get('askYield'),
            cached_ttm=d.get('ttm'),
            notes=d.get('notes', ''),
        )

    @property
    def ttm_years(self) -> float:
        """Time to maturity in years."""
        days = (self.maturity - date.today()).days
        return max(0, days / 365.25)

    @property
    def maturity_bucket(self) -> MaturityBucket:
        """Classify by time to maturity."""
        ttm = self.ttm_years
        if ttm <= 3:
            return MaturityBucket.SHORT
        elif ttm <= 7:
            return MaturityBucket.MEDIUM
        elif ttm <= 15:
            return MaturityBucket.LONG
        else:
            return MaturityBucket.ULTRA

    @property
    def is_matured(self) -> bool:
        """Check if sukuk has already matured."""
        return self.maturity < date.today()


@dataclass
class Quote:
    """
    Live market quote from IBKR.
    """
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    asof_ts: Optional[datetime] = None
    source: str = "IBKR"

    @property
    def mid(self) -> Optional[float]:
        """Mid price (bid/ask average)."""
        if self.bid and self.ask and self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last or self.close

    @property
    def spread_bps(self) -> Optional[float]:
        """Bid-ask spread in basis points."""
        if self.bid and self.ask and self.bid > 0 and self.ask > 0:
            mid = (self.bid + self.ask) / 2
            if mid > 0:
                return ((self.ask - self.bid) / mid) * 10000
        return None

    @property
    def stale_seconds(self) -> Optional[int]:
        """Seconds since quote was fetched."""
        if self.asof_ts:
            try:
                # Simple subtraction - both should be naive datetime
                return int((datetime.now() - self.asof_ts).total_seconds())
            except Exception:
                return None
        return None

    @property
    def has_valid_price(self) -> bool:
        """Check if we have any valid price."""
        return any([
            self.mid and self.mid > 0,
            self.last and self.last > 0,
            self.close and self.close > 0
        ])

    @property
    def best_price(self) -> Optional[float]:
        """Best available price (mid > last > close)."""
        if self.mid and self.mid > 0:
            return self.mid
        if self.last and self.last > 0:
            return self.last
        if self.close and self.close > 0:
            return self.close
        return None


@dataclass
class SukukLiveData:
    """
    Sukuk instrument with live market data.
    """
    instrument: SukukInstrument
    quote: Quote
    data_quality: DataQuality = DataQuality.OK
    warnings: List[str] = field(default_factory=list)

    @property
    def price_pct(self) -> Optional[float]:
        """Price as percentage of face value."""
        price = self.quote.best_price
        if price is None:
            return None
        # Bond prices are typically quoted as % of par
        # If > 200 or < 20, likely wrong convention
        if price > 200 or price < 20:
            return None  # Suspicious
        return price

    @property
    def bid_ask_bps(self) -> Optional[float]:
        """Bid-ask spread in basis points."""
        return self.quote.spread_bps

    @property
    def carry_proxy(self) -> float:
        """
        Carry proxy using coupon rate.
        Phase 1: Use coupon as proxy (NOT YTM - that requires more data).
        """
        return self.instrument.coupon_rate_pct

    @property
    def ask_yield(self) -> Optional[float]:
        """Ask yield if available from cache."""
        return self.instrument.cached_yield

    @property
    def ttm_years(self) -> float:
        """Time to maturity in years."""
        return self.instrument.ttm_years

    @classmethod
    def from_cached(cls, instrument: SukukInstrument) -> 'SukukLiveData':
        """
        Create SukukLiveData from cached data in instrument.

        Use when IBKR is not connected but we have cached prices from JSON.
        """
        quote = Quote(
            last=instrument.cached_price,
            close=instrument.cached_price,
            asof_ts=datetime.now(),
            source="CACHED"
        )

        live_data = cls(
            instrument=instrument,
            quote=quote,
            data_quality=DataQuality.STALE if instrument.cached_price else DataQuality.MISSING,
            warnings=["Using cached price from JSON (IBKR not connected)"] if instrument.cached_price else ["No cached price available"]
        )

        return live_data

    def validate(self) -> 'SukukLiveData':
        """Run validation checks and update data_quality."""
        self.warnings = []
        quality = DataQuality.OK

        # Check if matured
        if self.instrument.is_matured:
            self.warnings.append(f"MATURED on {self.instrument.maturity}")
            quality = DataQuality.MISSING

        # Check for missing price
        elif not self.quote.has_valid_price:
            self.warnings.append("No valid price available")
            quality = DataQuality.MISSING

        # Check if using cached data
        elif self.quote.source == "CACHED":
            self.warnings.append("Using cached price (IBKR not connected)")
            quality = DataQuality.STALE

        # Check staleness
        elif self.quote.stale_seconds and self.quote.stale_seconds > 3600:
            self.warnings.append(f"Stale quote: {self.quote.stale_seconds}s old")
            quality = DataQuality.STALE

        # Check bid > ask (inverted)
        elif self.quote.bid and self.quote.ask and self.quote.bid > self.quote.ask:
            self.warnings.append("Inverted bid/ask")
            quality = DataQuality.SUSPICIOUS

        # Check suspicious price range
        elif self.price_pct is None and self.quote.has_valid_price:
            self.warnings.append("Price outside normal range (20-200)")
            quality = DataQuality.SUSPICIOUS

        # Check wide spread
        elif self.bid_ask_bps and self.bid_ask_bps > 200:
            self.warnings.append(f"Wide spread: {self.bid_ask_bps:.0f} bps")
            quality = DataQuality.DEGRADED

        self.data_quality = quality
        return self


@dataclass
class SukukSignal:
    """
    Trading signal for a sukuk.
    """
    instrument: SukukInstrument
    live_data: Optional[SukukLiveData]
    action: SukukAction
    conviction: int  # 0-100
    size_cap_pct: float  # Maximum position size as % of portfolio
    reason: str
    data_quality: DataQuality

    # Details
    price: Optional[float] = None
    carry_proxy: Optional[float] = None
    ttm_years: Optional[float] = None
    bid_ask_bps: Optional[float] = None

    # Portfolio context
    current_weight_pct: float = 0.0
    issuer_exposure_pct: float = 0.0

    @classmethod
    def watch_signal(cls, instrument: SukukInstrument, reason: str,
                     live_data: Optional[SukukLiveData] = None) -> 'SukukSignal':
        """Create a WATCH signal (no action, monitoring only)."""
        return cls(
            instrument=instrument,
            live_data=live_data,
            action=SukukAction.WATCH,
            conviction=0,
            size_cap_pct=0,
            reason=reason,
            data_quality=DataQuality.DEGRADED if live_data else DataQuality.MISSING,
            price=live_data.price_pct if live_data else None,
            carry_proxy=live_data.carry_proxy if live_data else instrument.coupon_rate_pct,
            ttm_years=instrument.ttm_years,
            bid_ask_bps=live_data.bid_ask_bps if live_data else None,
        )


@dataclass
class RiskLimits:
    """Portfolio risk limits for sukuk."""
    max_single_position_pct: float = 5.0
    max_issuer_pct: float = 15.0
    max_total_sukuk_pct: float = 30.0
    min_bid_ask_bps: float = 200  # Max acceptable spread
    max_stale_seconds: int = 3600  # 1 hour

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RiskLimits':
        return cls(
            max_single_position_pct=d.get('max_single_position_pct', 5.0),
            max_issuer_pct=d.get('max_issuer_pct', 15.0),
            max_total_sukuk_pct=d.get('max_total_sukuk_pct', 30.0),
            min_bid_ask_bps=d.get('min_bid_ask_bps', 200),
            max_stale_seconds=d.get('max_stale_seconds', 3600),
        )


@dataclass
@dataclass
class SukukUniverse:
    """
    Complete sukuk universe with all instruments and limits.
    """
    instruments: List[SukukInstrument]
    risk_limits: RiskLimits
    issuer_mapping: Dict[str, Dict] = field(default_factory=dict)
    version: str = "1.0"
    updated: str = ""

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'SukukUniverse':
        """Load from JSON config file."""
        metadata = data.get('metadata', {})
        risk_limits = RiskLimits.from_dict(data.get('risk_limits', {}))
        issuer_mapping = data.get('issuer_mapping', {})

        instruments = [
            SukukInstrument.from_dict(s, issuer_mapping)
            for s in data.get('sukuk', [])
        ]

        return cls(
            instruments=instruments,
            risk_limits=risk_limits,
            issuer_mapping=issuer_mapping,
            version=metadata.get('version', '1.0'),
            updated=metadata.get('updated', ''),
        )

    @classmethod
    def from_hysk_json(cls, sukuk_list: List[Dict],
                       issuer_mapping: Dict[str, Dict] = None,
                       risk_limits: RiskLimits = None) -> 'SukukUniverse':
        """
        Load directly from HYSK-holdings_ibkr.json format.

        Args:
            sukuk_list: List of sukuk dicts from HYSK JSON
            issuer_mapping: Optional issuer classification
            risk_limits: Optional risk limits (uses defaults if not provided)
        """
        if issuer_mapping is None:
            issuer_mapping = DEFAULT_ISSUER_MAPPING

        if risk_limits is None:
            risk_limits = RiskLimits()

        instruments = [
            SukukInstrument.from_dict(s, issuer_mapping)
            for s in sukuk_list
        ]

        return cls(
            instruments=instruments,
            risk_limits=risk_limits,
            issuer_mapping=issuer_mapping,
        )

    @property
    def active_instruments(self) -> List[SukukInstrument]:
        """Get only non-matured instruments with valid conid."""
        return [
            i for i in self.instruments
            if not i.is_matured and i.conid > 0
        ]

    def get_by_conid(self, conid: int) -> Optional[SukukInstrument]:
        """Find instrument by conid."""
        for i in self.instruments:
            if i.conid == conid:
                return i
        return None

    def get_by_isin(self, isin: str) -> Optional[SukukInstrument]:
        """Find instrument by ISIN."""
        for i in self.instruments:
            if i.isin == isin:
                return i
        return None

    def issuer_buckets(self) -> Dict[str, List[SukukInstrument]]:
        """Group instruments by issuer bucket."""
        buckets: Dict[str, List[SukukInstrument]] = {}
        for i in self.instruments:
            if i.issuer_bucket not in buckets:
                buckets[i.issuer_bucket] = []
            buckets[i.issuer_bucket].append(i)
        return buckets