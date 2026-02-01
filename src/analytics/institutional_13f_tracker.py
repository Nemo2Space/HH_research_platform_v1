"""
13F Institutional Holdings Tracker

Tracks what major hedge funds and institutions are buying/selling.
Data from SEC 13F filings (required for funds with >$100M AUM).

IMPORTANT LIMITATIONS:
- 45+ day lag (filed 45 days after quarter end)
- Quarterly snapshots only (not real-time)
- May have already sold by time filing is public
- Best used for: long-term conviction, new position alerts, concentration analysis

Notable Filers Tracked:
- Berkshire Hathaway (Warren Buffett)
- Bridgewater Associates (Ray Dalio)
- Pershing Square (Bill Ackman)
- Appaloosa Management (David Tepper)
- Greenlight Capital (David Einhorn)
- Tiger Global
- Citadel
- And many more...

Author: Alpha Research Platform
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import xml.etree.ElementTree as ET
import re
import time
import os
import json

from src.utils.logging import get_logger

logger = get_logger(__name__)


# SEC requires user agent
SEC_USER_AGENT = os.getenv('SEC_USER_AGENT', 'AlphaResearch contact@example.com')
SEC_HEADERS = {
    'User-Agent': SEC_USER_AGENT,
    'Accept-Encoding': 'gzip, deflate',
}

# Notable institutions to track (CIK numbers)
NOTABLE_INSTITUTIONS = {
    '0001067983': {'name': 'Berkshire Hathaway', 'manager': 'Warren Buffett', 'style': 'Value'},
    '0001350694': {'name': 'Bridgewater Associates', 'manager': 'Ray Dalio', 'style': 'Macro'},
    '0001336528': {'name': 'Pershing Square', 'manager': 'Bill Ackman', 'style': 'Activist'},
    '0001656456': {'name': 'Appaloosa Management', 'manager': 'David Tepper', 'style': 'Event-Driven'},
    '0001079114': {'name': 'Greenlight Capital', 'manager': 'David Einhorn', 'style': 'Value'},
    '0001167483': {'name': 'Tiger Global', 'manager': 'Chase Coleman', 'style': 'Growth'},
    '0001423053': {'name': 'Citadel Advisors', 'manager': 'Ken Griffin', 'style': 'Quant'},
    '0001037389': {'name': 'Renaissance Technologies', 'manager': 'Jim Simons', 'style': 'Quant'},
    '0001061768': {'name': 'Elliott Management', 'manager': 'Paul Singer', 'style': 'Activist'},
    '0001029160': {'name': 'Third Point', 'manager': 'Dan Loeb', 'style': 'Activist'},
    '0001040273': {'name': 'ValueAct Capital', 'manager': 'Jeff Ubben', 'style': 'Activist'},
    '0001103804': {'name': 'Baupost Group', 'manager': 'Seth Klarman', 'style': 'Value'},
    '0001541617': {'name': 'Coatue Management', 'manager': 'Philippe Laffont', 'style': 'Tech'},
    '0001649339': {'name': 'Point72', 'manager': 'Steve Cohen', 'style': 'Multi-Strategy'},
    '0001510476': {'name': 'Viking Global', 'manager': 'Andreas Halvorsen', 'style': 'Long/Short'},
}


class ChangeType(Enum):
    """Type of position change."""
    NEW = "NEW"           # New position
    ADDED = "ADDED"       # Increased position
    REDUCED = "REDUCED"   # Decreased position
    SOLD = "SOLD"         # Completely sold
    UNCHANGED = "UNCHANGED"


@dataclass
class HoldingPosition:
    """Single holding in a 13F filing."""
    ticker: str
    company_name: str
    cusip: str

    # Current quarter
    shares: int
    value: float  # In thousands (as reported)

    # Previous quarter (for comparison)
    prev_shares: int = 0
    prev_value: float = 0

    # Change
    shares_change: int = 0
    shares_change_pct: float = 0
    change_type: ChangeType = ChangeType.UNCHANGED

    # Weight in portfolio
    portfolio_weight_pct: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'company_name': self.company_name,
            'shares': self.shares,
            'value': self.value,
            'shares_change': self.shares_change,
            'shares_change_pct': self.shares_change_pct,
            'change_type': self.change_type.value,
            'portfolio_weight_pct': self.portfolio_weight_pct,
        }


@dataclass
class InstitutionalFiling:
    """A single 13F filing from an institution."""
    cik: str
    institution_name: str
    manager_name: str
    investment_style: str

    # Filing info
    filing_date: date
    report_date: date  # Quarter end date

    # Aggregate stats
    total_value: float  # In thousands
    num_positions: int

    # Holdings
    holdings: List[HoldingPosition] = field(default_factory=list)

    # Top positions
    top_holdings: List[str] = field(default_factory=list)  # Top 10 tickers

    # Changes from previous quarter
    new_positions: List[str] = field(default_factory=list)
    increased_positions: List[str] = field(default_factory=list)
    reduced_positions: List[str] = field(default_factory=list)
    sold_positions: List[str] = field(default_factory=list)


@dataclass
class TickerInstitutionalOwnership:
    """Institutional ownership data for a single ticker."""
    ticker: str
    as_of_date: date

    # Aggregate ownership
    num_institutions: int = 0
    total_shares: int = 0
    total_value: float = 0

    # Notable holders
    notable_holders: List[Dict] = field(default_factory=list)

    # Recent changes
    new_buyers: List[str] = field(default_factory=list)      # Institutions that initiated
    added_by: List[str] = field(default_factory=list)        # Institutions that added
    reduced_by: List[str] = field(default_factory=list)      # Institutions that reduced
    sold_by: List[str] = field(default_factory=list)         # Institutions that sold

    # Signal
    signal: str = "NEUTRAL"
    signal_strength: int = 50

    # Key metrics
    buffett_owns: bool = False
    buffett_added: bool = False
    activist_involved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticker': self.ticker,
            'num_institutions': self.num_institutions,
            'total_shares': self.total_shares,
            'new_buyers': self.new_buyers,
            'added_by': self.added_by,
            'reduced_by': self.reduced_by,
            'sold_by': self.sold_by,
            'signal': self.signal,
            'signal_strength': self.signal_strength,
            'buffett_owns': self.buffett_owns,
        }


class Institutional13FTracker:
    """
    Tracks 13F institutional holdings from SEC EDGAR.
    """

    def __init__(self):
        self.base_url = "https://www.sec.gov"
        self._cache = {}
        self._holdings_cache = {}
        self._cache_duration = timedelta(hours=24)  # 13F data is stale anyway
        self._last_request = 0
        self._rate_limit = 0.15

        # CUSIP to ticker mapping cache
        self._cusip_map = {}

    def _rate_limited_get(self, url: str) -> Optional[requests.Response]:
        """Make rate-limited request to SEC."""
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)

        try:
            response = requests.get(url, headers=SEC_HEADERS, timeout=30)
            self._last_request = time.time()

            if response.status_code == 200:
                return response
            else:
                logger.warning(f"SEC request failed: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"SEC request error: {e}")
            return None

    def get_institution_holdings(self, cik: str) -> Optional[InstitutionalFiling]:
        """
        Get latest 13F holdings for an institution.

        Args:
            cik: CIK number (e.g., '0001067983' for Berkshire)

        Returns:
            InstitutionalFiling with all holdings
        """
        # Normalize CIK
        cik = cik.zfill(10)

        # Check cache
        if cik in self._cache:
            cached_time, cached_data = self._cache[cik]
            if datetime.now() - cached_time < self._cache_duration:
                return cached_data

        try:
            # Get recent filings
            filings = self._get_13f_filings(cik, limit=2)

            if not filings:
                logger.warning(f"No 13F filings found for CIK {cik}")
                return None

            # Parse latest and previous filings
            latest = self._parse_13f_filing(filings[0], cik)

            if latest and len(filings) > 1:
                previous = self._parse_13f_filing(filings[1], cik)
                if previous:
                    self._calculate_changes(latest, previous)

            # Cache result
            self._cache[cik] = (datetime.now(), latest)

            return latest

        except Exception as e:
            logger.error(f"Error getting holdings for {cik}: {e}")
            return None

    def get_notable_institutions_holding(self, ticker: str) -> TickerInstitutionalOwnership:
        """
        Get which notable institutions hold a specific ticker.

        Args:
            ticker: Stock symbol

        Returns:
            TickerInstitutionalOwnership with institutional ownership data
        """
        result = TickerInstitutionalOwnership(
            ticker=ticker.upper(),
            as_of_date=date.today(),
        )

        notable_holders = []

        for cik, info in NOTABLE_INSTITUTIONS.items():
            try:
                filing = self.get_institution_holdings(cik)

                if not filing:
                    continue

                # Check if they hold this ticker
                for holding in filing.holdings:
                    if holding.ticker.upper() == ticker.upper():
                        result.num_institutions += 1
                        result.total_shares += holding.shares
                        result.total_value += holding.value

                        holder_info = {
                            'name': info['name'],
                            'manager': info['manager'],
                            'style': info['style'],
                            'shares': holding.shares,
                            'value': holding.value,
                            'weight': holding.portfolio_weight_pct,
                            'change': holding.change_type.value,
                            'change_pct': holding.shares_change_pct,
                        }
                        notable_holders.append(holder_info)

                        # Track changes
                        if holding.change_type == ChangeType.NEW:
                            result.new_buyers.append(info['name'])
                        elif holding.change_type == ChangeType.ADDED:
                            result.added_by.append(info['name'])
                        elif holding.change_type == ChangeType.REDUCED:
                            result.reduced_by.append(info['name'])
                        elif holding.change_type == ChangeType.SOLD:
                            result.sold_by.append(info['name'])

                        # Special flags
                        if info['name'] == 'Berkshire Hathaway':
                            result.buffett_owns = True
                            if holding.change_type in [ChangeType.NEW, ChangeType.ADDED]:
                                result.buffett_added = True

                        if info['style'] == 'Activist':
                            result.activist_involved = True

                        break

            except Exception as e:
                logger.debug(f"Error checking {info['name']}: {e}")
                continue

        result.notable_holders = notable_holders

        # Calculate signal
        result.signal, result.signal_strength = self._calculate_signal(result)

        return result

    def get_buffett_holdings(self) -> Optional[InstitutionalFiling]:
        """Get Warren Buffett / Berkshire Hathaway holdings."""
        return self.get_institution_holdings('0001067983')

    def get_all_notable_holdings(self) -> Dict[str, InstitutionalFiling]:
        """Get holdings for all notable institutions."""
        results = {}

        for cik, info in NOTABLE_INSTITUTIONS.items():
            try:
                filing = self.get_institution_holdings(cik)
                if filing:
                    results[info['name']] = filing
            except Exception as e:
                logger.debug(f"Error getting {info['name']}: {e}")

        return results

    def _calculate_signal(self, ownership: TickerInstitutionalOwnership) -> Tuple[str, int]:
        """Calculate signal from institutional ownership."""
        score = 50

        # Buffett factor
        if ownership.buffett_added:
            score += 20
        elif ownership.buffett_owns:
            score += 10

        # New buyers
        num_new = len(ownership.new_buyers)
        if num_new >= 3:
            score += 15
        elif num_new >= 1:
            score += 8

        # Net adding vs reducing
        num_adding = len(ownership.added_by)
        num_reducing = len(ownership.reduced_by)
        num_selling = len(ownership.sold_by)

        net_change = num_adding - num_reducing - (num_selling * 2)

        if net_change >= 3:
            score += 15
        elif net_change >= 1:
            score += 8
        elif net_change <= -3:
            score -= 15
        elif net_change <= -1:
            score -= 8

        # Activist involvement (could be positive or negative)
        if ownership.activist_involved:
            score += 5  # Generally bullish - they see value

        # Number of notable holders
        if ownership.num_institutions >= 5:
            score += 10
        elif ownership.num_institutions >= 3:
            score += 5

        # Clamp
        score = max(0, min(100, score))

        if score >= 75:
            return "STRONG_BUY", score
        elif score >= 60:
            return "BUY", score
        elif score <= 25:
            return "STRONG_SELL", score
        elif score <= 40:
            return "SELL", score
        else:
            return "NEUTRAL", score

    def _get_13f_filings(self, cik: str, limit: int = 2) -> List[Dict]:
        """Get list of 13F filings for a CIK."""
        filings = []

        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = self._rate_limited_get(url)

            if not response:
                return filings

            data = response.json()
            recent = data.get('filings', {}).get('recent', {})

            forms = recent.get('form', [])
            dates = recent.get('filingDate', [])
            accessions = recent.get('accessionNumber', [])
            report_dates = recent.get('reportDate', [])

            count = 0
            for i, form in enumerate(forms):
                if form == '13F-HR' and count < limit:
                    filings.append({
                        'cik': cik,
                        'accession': accessions[i].replace('-', ''),
                        'filing_date': datetime.strptime(dates[i], '%Y-%m-%d').date(),
                        'report_date': datetime.strptime(report_dates[i], '%Y-%m-%d').date() if report_dates[i] else None,
                    })
                    count += 1

        except Exception as e:
            logger.error(f"Error getting 13F filings: {e}")

        return filings

    def _parse_13f_filing(self, filing: Dict, cik: str) -> Optional[InstitutionalFiling]:
        """Parse a 13F filing."""
        try:
            accession = filing['accession']

            # Get institution info
            inst_info = NOTABLE_INSTITUTIONS.get(cik, {
                'name': 'Unknown',
                'manager': 'Unknown',
                'style': 'Unknown'
            })

            result = InstitutionalFiling(
                cik=cik,
                institution_name=inst_info.get('name', 'Unknown'),
                manager_name=inst_info.get('manager', 'Unknown'),
                investment_style=inst_info.get('style', 'Unknown'),
                filing_date=filing['filing_date'],
                report_date=filing.get('report_date') or filing['filing_date'],
                total_value=0,
                num_positions=0,
            )

            # Find and parse the information table
            url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}"
            index_url = f"{url}/index.json"

            response = self._rate_limited_get(index_url)
            if not response:
                return result

            index_data = response.json()

            # Find infotable.xml
            xml_file = None
            for item in index_data.get('directory', {}).get('item', []):
                name = item.get('name', '').lower()
                if 'infotable' in name and name.endswith('.xml'):
                    xml_file = item.get('name')
                    break

            if not xml_file:
                # Try to find any XML that might contain holdings
                for item in index_data.get('directory', {}).get('item', []):
                    name = item.get('name', '').lower()
                    if name.endswith('.xml') and 'primary' not in name:
                        xml_file = item.get('name')
                        break

            if not xml_file:
                logger.warning(f"No infotable found for {accession}")
                return result

            # Parse holdings
            xml_url = f"{url}/{xml_file}"
            response = self._rate_limited_get(xml_url)

            if not response:
                return result

            holdings = self._parse_infotable(response.content)
            result.holdings = holdings
            result.num_positions = len(holdings)
            result.total_value = sum(h.value for h in holdings)

            # Calculate weights and top holdings
            for h in holdings:
                if result.total_value > 0:
                    h.portfolio_weight_pct = (h.value / result.total_value) * 100

            # Sort by value and get top holdings
            holdings.sort(key=lambda x: x.value, reverse=True)
            result.top_holdings = [h.ticker for h in holdings[:10] if h.ticker]

            return result

        except Exception as e:
            logger.error(f"Error parsing 13F filing: {e}")
            return None

    def _parse_infotable(self, xml_content: bytes) -> List[HoldingPosition]:
        """Parse the infotable XML to extract holdings."""
        holdings = []

        try:
            content = xml_content.decode('utf-8', errors='ignore')

            # More aggressive namespace removal for SEC XML files
            # Remove XML declaration if present
            content = re.sub(r'<\?xml[^>]*\?>', '', content)

            # Remove all namespace declarations
            content = re.sub(r'\sxmlns[^=]*="[^"]*"', '', content)

            # Remove namespace prefixes from tags (e.g., ns1:infoTable -> infoTable)
            content = re.sub(r'<(/?)[\w]+:', r'<\1', content)

            # Try to parse
            try:
                root = ET.fromstring(content)
            except ET.ParseError:
                # If still failing, try wrapping in a root element
                content = f"<root>{content}</root>"
                try:
                    root = ET.fromstring(content)
                except ET.ParseError as e:
                    logger.debug(f"XML parse failed even with wrapper: {e}")
                    return holdings

            # Find all infoTable entries
            for entry in root.iter():
                tag_lower = entry.tag.lower() if entry.tag else ""
                if 'infotable' in tag_lower and tag_lower != 'informationtable':
                    try:
                        holding = self._parse_holding_entry(entry)
                        if holding and holding.shares > 0:
                            holdings.append(holding)
                    except Exception as e:
                        logger.debug(f"Error parsing entry: {e}")

        except Exception as e:
            logger.debug(f"Error parsing infotable: {e}")

        return holdings

    def _parse_holding_entry(self, entry) -> Optional[HoldingPosition]:
        """Parse a single holding entry."""
        try:
            # Get elements - handle different possible structures
            name = self._get_element_text(entry, ['nameOfIssuer', 'name'])
            cusip = self._get_element_text(entry, ['cusip'])
            value = self._get_element_text(entry, ['value'])
            shares = self._get_element_text(entry, ['sshPrnamt', 'shares', 'shrsOrPrnAmt/sshPrnamt'])

            if not cusip or not value:
                return None

            # Convert to ticker
            ticker = self._cusip_to_ticker(cusip, name)

            return HoldingPosition(
                ticker=ticker or cusip[:8],  # Use CUSIP if ticker not found
                company_name=name or "",
                cusip=cusip,
                shares=int(float(shares)) if shares else 0,
                value=float(value) if value else 0,
            )

        except Exception as e:
            logger.debug(f"Holding entry parse error: {e}")
            return None

    def _get_element_text(self, parent, names: List[str]) -> Optional[str]:
        """Get text from element trying multiple possible names."""
        for name in names:
            if '/' in name:
                # Handle nested path
                parts = name.split('/')
                elem = parent
                for part in parts:
                    found = None
                    for child in elem:
                        if part.lower() in child.tag.lower():
                            found = child
                            break
                    if found is not None:
                        elem = found
                    else:
                        break
                else:
                    if elem.text:
                        return elem.text.strip()
            else:
                for elem in parent.iter():
                    if name.lower() in elem.tag.lower():
                        if elem.text:
                            return elem.text.strip()
        return None

    def _cusip_to_ticker(self, cusip: str, name: str = None) -> Optional[str]:
        """Convert CUSIP to ticker symbol."""
        # Check cache first
        if cusip in self._cusip_map:
            return self._cusip_map[cusip]

        # Direct CUSIP mappings for common holdings (8-char CUSIP without check digit)
        cusip_direct = {
            '03783310': 'AAPL',    # Apple
            '59491810': 'MSFT',    # Microsoft
            '02313510': 'AMZN',    # Amazon
            '30303M10': 'META',    # Meta
            '88160R10': 'TSLA',    # Tesla
            '67066G10': 'NVDA',    # Nvidia
            '02079K30': 'GOOGL',   # Alphabet C
            '02079K10': 'GOOG',    # Alphabet A
            '084670702': 'BRK.B',  # Berkshire B
            '46625H10': 'JPM',     # JPMorgan
            '92826C83': 'V',       # Visa
            '57636Q10': 'MA',      # Mastercard
            '91324P10': 'UNH',     # UnitedHealth
            '437076102': 'HD',     # Home Depot
            '166764100': 'CVX',    # Chevron
            '191216100': 'KO',     # Coca-Cola
            '00287Y109': 'ABBV',   # AbbVie
            '717081103': 'PFE',    # Pfizer
            '30231G102': 'XOM',    # Exxon
            '22160K105': 'COST',   # Costco
            '254687106': 'DIS',    # Disney
            '64110L106': 'NFLX',   # Netflix
            '00724F101': 'ADBE',   # Adobe
            '79466L302': 'CRM',    # Salesforce
            '458140100': 'INTC',   # Intel
            '17275R102': 'CSCO',   # Cisco
            '92343V104': 'VZ',     # Verizon
            '20030N101': 'CMCSA',  # Comcast
            '931142103': 'WMT',    # Walmart
            '060505104': 'BAC',    # Bank of America
            '949746101': 'WFC',    # Wells Fargo
            '38141G104': 'GS',     # Goldman Sachs
            '617446448': 'MS',     # Morgan Stanley
            '025816109': 'AXP',    # American Express
            '149123101': 'CAT',    # Caterpillar
            '097023105': 'BA',     # Boeing
            '580135101': 'MCD',    # McDonald's
            '654106103': 'NKE',    # Nike
            '855244109': 'SBUX',   # Starbucks
            '478160104': 'JNJ',    # Johnson & Johnson
            '742718109': 'PG',     # Procter & Gamble
            '60871R209': 'MRK',    # Merck
            '00206R102': 'T',      # AT&T
            '500754106': 'KHC',    # Kraft Heinz
            '69331C108': 'PM',     # Philip Morris
            '693475105': 'OXY',    # Occidental Petroleum
            '58933Y105': 'MCO',    # Moody's
            '23918K108': 'DVA',    # DaVita
            '444859102': 'HPQ',    # HP
            '532457108': 'LLY',    # Eli Lilly
            '02209S103': 'ALLY',   # Ally Financial
            'H1467J10': 'HEI',     # HEICO
            '422806109': 'HEI',    # HEICO (alternate)
            '25470M109': 'DPZ',    # Domino's Pizza
            '73278L105': 'POOL',   # Pool Corp
            '90384S303': 'ULTA',   # Ulta Beauty
            'H1467J104': 'HEI.A',  # HEICO Class A
        }

        # Try direct CUSIP lookup (first 8 chars)
        cusip_key = cusip[:8] if len(cusip) >= 8 else cusip
        if cusip_key in cusip_direct:
            self._cusip_map[cusip] = cusip_direct[cusip_key]
            return cusip_direct[cusip_key]

        # Also try full CUSIP
        if cusip in cusip_direct:
            self._cusip_map[cusip] = cusip_direct[cusip]
            return cusip_direct[cusip]

        # Try to derive from name
        if name:
            name_upper = name.upper()

            # Well-known companies by name
            known_mappings = {
                'APPLE': 'AAPL',
                'MICROSOFT': 'MSFT',
                'AMAZON': 'AMZN',
                'ALPHABET': 'GOOGL',
                'META PLATFORMS': 'META',
                'FACEBOOK': 'META',
                'TESLA': 'TSLA',
                'NVIDIA': 'NVDA',
                'BERKSHIRE': 'BRK.B',
                'JPMORGAN': 'JPM',
                'JP MORGAN': 'JPM',
                'JOHNSON & JOHNSON': 'JNJ',
                'JOHNSON AND JOHNSON': 'JNJ',
                'VISA': 'V',
                'PROCTER': 'PG',
                'MASTERCARD': 'MA',
                'UNITEDHEALTH': 'UNH',
                'HOME DEPOT': 'HD',
                'CHEVRON': 'CVX',
                'COCA-COLA': 'KO',
                'COCA COLA': 'KO',
                'ABBVIE': 'ABBV',
                'PFIZER': 'PFE',
                'EXXON': 'XOM',
                'COSTCO': 'COST',
                'DISNEY': 'DIS',
                'WALT DISNEY': 'DIS',
                'NETFLIX': 'NFLX',
                'ADOBE': 'ADBE',
                'SALESFORCE': 'CRM',
                'INTEL': 'INTC',
                'CISCO': 'CSCO',
                'VERIZON': 'VZ',
                'COMCAST': 'CMCSA',
                'WALMART': 'WMT',
                'WAL-MART': 'WMT',
                'BANK OF AMERICA': 'BAC',
                'WELLS FARGO': 'WFC',
                'GOLDMAN': 'GS',
                'MORGAN STANLEY': 'MS',
                'AMERICAN EXPRESS': 'AXP',
                'CATERPILLAR': 'CAT',
                'BOEING': 'BA',
                'MCDONALD': 'MCD',
                'NIKE': 'NKE',
                'STARBUCKS': 'SBUX',
                'KRAFT HEINZ': 'KHC',
                'PHILIP MORRIS': 'PM',
                'OCCIDENTAL': 'OXY',
                'MOODY': 'MCO',
                'DAVITA': 'DVA',
                'ELI LILLY': 'LLY',
                'LILLY': 'LLY',
                'ALLY FINANCIAL': 'ALLY',
                'MERCK': 'MRK',
                'AT&T': 'T',
                'KROGER': 'KR',
                'GENERAL MOTORS': 'GM',
                'FORD': 'F',
                'PARAMOUNT': 'PARA',
                'SIRIUS': 'SIRI',
                'LIBERTY': 'LSXM',
                'CHARTER': 'CHTR',
                'VERISIGN': 'VRSN',
                'SNOWFLAKE': 'SNOW',
                'NU HOLDINGS': 'NU',
                'CAPITAL ONE': 'COF',
                'CITIGROUP': 'C',
                'CITI': 'C',
                'HEICO': 'HEI',
                'DOMINO': 'DPZ',
                'POOL CORP': 'POOL',
                'ULTA': 'ULTA',
                'CHUBB': 'CB',
                'T-MOBILE': 'TMUS',
                'VERISIGN': 'VRSN',
            }

            for key, ticker in known_mappings.items():
                if key in name_upper:
                    self._cusip_map[cusip] = ticker
                    return ticker

        # Return None if not found (will show CUSIP instead)
        return None

    def _calculate_changes(self, current: InstitutionalFiling,
                           previous: InstitutionalFiling) -> None:
        """Calculate changes between two filings."""
        # Build previous holdings map
        prev_map = {h.cusip: h for h in previous.holdings}
        curr_map = {h.cusip: h for h in current.holdings}

        # Find changes
        for holding in current.holdings:
            if holding.cusip in prev_map:
                prev = prev_map[holding.cusip]
                holding.prev_shares = prev.shares
                holding.prev_value = prev.value
                holding.shares_change = holding.shares - prev.shares

                if prev.shares > 0:
                    holding.shares_change_pct = ((holding.shares - prev.shares) / prev.shares) * 100

                if holding.shares_change > 0:
                    holding.change_type = ChangeType.ADDED
                    current.increased_positions.append(holding.ticker)
                elif holding.shares_change < 0:
                    holding.change_type = ChangeType.REDUCED
                    current.reduced_positions.append(holding.ticker)
                else:
                    holding.change_type = ChangeType.UNCHANGED
            else:
                # New position
                holding.change_type = ChangeType.NEW
                holding.shares_change = holding.shares
                holding.shares_change_pct = 100
                current.new_positions.append(holding.ticker)

        # Find sold positions
        for cusip, prev_holding in prev_map.items():
            if cusip not in curr_map:
                current.sold_positions.append(prev_holding.ticker)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_tracker = None

def get_13f_tracker() -> Institutional13FTracker:
    """Get singleton tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = Institutional13FTracker()
    return _tracker


def get_institutional_ownership(ticker: str) -> TickerInstitutionalOwnership:
    """
    Get institutional ownership for a ticker from notable funds.

    Usage:
        ownership = get_institutional_ownership('AAPL')
        print(f"Notable holders: {ownership.num_institutions}")
        print(f"Buffett owns: {ownership.buffett_owns}")
        print(f"New buyers: {ownership.new_buyers}")
    """
    tracker = get_13f_tracker()
    return tracker.get_notable_institutions_holding(ticker)


def get_buffett_portfolio() -> Optional[InstitutionalFiling]:
    """
    Get Warren Buffett's current portfolio.

    Usage:
        portfolio = get_buffett_portfolio()
        print(f"Top holdings: {portfolio.top_holdings}")
        print(f"New positions: {portfolio.new_positions}")
    """
    tracker = get_13f_tracker()
    return tracker.get_buffett_holdings()


def get_institution_portfolio(name: str) -> Optional[InstitutionalFiling]:
    """
    Get portfolio for a specific institution by name.

    Usage:
        portfolio = get_institution_portfolio('Pershing Square')
        print(f"Top holdings: {portfolio.top_holdings}")
    """
    tracker = get_13f_tracker()

    # Find CIK by name
    for cik, info in NOTABLE_INSTITUTIONS.items():
        if name.lower() in info['name'].lower():
            return tracker.get_institution_holdings(cik)

    return None


def get_institutional_summary_table(tickers: List[str]) -> pd.DataFrame:
    """
    Get institutional ownership summary for multiple tickers.

    Usage:
        df = get_institutional_summary_table(['AAPL', 'MSFT', 'NVDA'])
        print(df)
    """
    tracker = get_13f_tracker()

    data = []
    for ticker in tickers:
        try:
            ownership = tracker.get_notable_institutions_holding(ticker)
            data.append({
                'Ticker': ticker,
                'Signal': ownership.signal,
                'Strength': ownership.signal_strength,
                '# Notable': ownership.num_institutions,
                'Buffett': '✅' if ownership.buffett_owns else '',
                'New Buyers': ', '.join(ownership.new_buyers[:2]) if ownership.new_buyers else '',
                'Added': ', '.join(ownership.added_by[:2]) if ownership.added_by else '',
                'Reduced': ', '.join(ownership.reduced_by[:2]) if ownership.reduced_by else '',
            })
        except Exception as e:
            logger.debug(f"{ticker}: 13F error: {e}")

    return pd.DataFrame(data)


def screen_for_institutional_buying(tickers: List[str],
                                     min_strength: int = 60) -> List[TickerInstitutionalOwnership]:
    """
    Screen for stocks with institutional buying.

    Usage:
        signals = screen_for_institutional_buying(universe, min_strength=70)
        for s in signals:
            print(f"{s.ticker}: New buyers: {s.new_buyers}")
    """
    tracker = get_13f_tracker()

    results = []
    for ticker in tickers:
        try:
            ownership = tracker.get_notable_institutions_holding(ticker)
            if ownership.signal_strength >= min_strength:
                results.append(ownership)
        except Exception:
            continue

    results.sort(key=lambda x: x.signal_strength, reverse=True)
    return results


def get_latest_moves_summary() -> pd.DataFrame:
    """
    Get summary of latest moves from all notable institutions.

    Returns DataFrame with new positions and significant changes.
    """
    tracker = get_13f_tracker()

    all_moves = []

    for cik, info in NOTABLE_INSTITUTIONS.items():
        try:
            filing = tracker.get_institution_holdings(cik)
            if not filing:
                continue

            # New positions
            for ticker in filing.new_positions:
                all_moves.append({
                    'Institution': info['name'],
                    'Manager': info['manager'],
                    'Ticker': ticker,
                    'Action': 'NEW',
                    'Style': info['style'],
                })

            # Significant adds (top 5)
            for ticker in filing.increased_positions[:5]:
                all_moves.append({
                    'Institution': info['name'],
                    'Manager': info['manager'],
                    'Ticker': ticker,
                    'Action': 'ADDED',
                    'Style': info['style'],
                })

            # Sold positions (interesting to track)
            for ticker in filing.sold_positions[:3]:
                all_moves.append({
                    'Institution': info['name'],
                    'Manager': info['manager'],
                    'Ticker': ticker,
                    'Action': 'SOLD',
                    'Style': info['style'],
                })

        except Exception as e:
            logger.debug(f"Error getting {info['name']}: {e}")

    return pd.DataFrame(all_moves)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing 13F Institutional Tracker...")

    # Test Buffett portfolio
    print("\n=== Buffett Portfolio ===")
    portfolio = get_buffett_portfolio()

    if portfolio:
        print(f"Filing Date: {portfolio.filing_date}")
        print(f"Total Value: ${portfolio.total_value/1000:.1f}B")
        print(f"Positions: {portfolio.num_positions}")
        print(f"Top 10: {portfolio.top_holdings}")
        print(f"New Positions: {portfolio.new_positions}")
        print(f"Increased: {portfolio.increased_positions[:5]}")

    # Test single ticker
    print("\n=== AAPL Institutional Ownership ===")
    ownership = get_institutional_ownership('AAPL')

    print(f"Notable Holders: {ownership.num_institutions}")
    print(f"Buffett Owns: {ownership.buffett_owns}")
    print(f"New Buyers: {ownership.new_buyers}")
    print(f"Added By: {ownership.added_by}")
    print(f"Signal: {ownership.signal} ({ownership.signal_strength})")