"""
IBKR FlexQuery CSV Parser
=========================

Parses IBKR Activity FlexQuery CSV files to extract:
- Daily NAV history
- Deposits/Withdrawals
- Dividends, Interest, Fees
- Position data
- Performance calculations (TWR, MWR)

Author: HH Research Platform
"""

import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import io


@dataclass
class CashTransaction:
    date: str
    amount: float
    type: str
    description: str
    symbol: str = ""


@dataclass
class DailyNAV:
    date: str
    cash: float
    stock: float
    bonds: float
    options: float
    dividends_accrual: float
    interest_accrual: float
    total: float


@dataclass
class NAVChange:
    date: str
    starting_value: float
    ending_value: float
    mtm: float
    realized: float
    unrealized_change: float
    deposits_withdrawals: float
    dividends: float
    interest: float
    commissions: float
    fees: float
    other: float


@dataclass
class Position:
    date: str
    symbol: str
    description: str
    quantity: float
    mark_price: float
    position_value: float
    cost_basis: float
    unrealized_pnl: float
    percent_of_nav: float


class IBKRFlexQueryParser:
    """Parser for IBKR FlexQuery CSV files."""

    def __init__(self, filepath: str, account_id: Optional[str] = None):
        self.filepath = filepath
        self.account_id = account_id

        # Parsed data
        self.accounts: List[str] = []
        self.nav_history: List[DailyNAV] = []
        self.nav_changes: List[NAVChange] = []
        self.cash_transactions: List[CashTransaction] = []
        self.positions: Dict[str, List[Position]] = {}  # date -> positions

        # Summary
        self.total_deposits = 0.0
        self.total_withdrawals = 0.0
        self.total_dividends = 0.0
        self.total_interest = 0.0
        self.total_fees = 0.0
        self.total_commissions = 0.0

    def parse(self) -> Dict[str, Any]:
        """Parse the FlexQuery CSV file."""

        with open(self.filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')

        current_section = None
        section_headers = []
        current_account = None

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # Parse CSV row
            try:
                reader = csv.reader(io.StringIO(line))
                row = next(reader)
            except:
                i += 1
                continue

            if not row:
                i += 1
                continue

            # Check for section markers
            if row[0] == 'BOF':
                # Beginning of file - extract account info
                pass
            elif row[0] == 'BOA':
                # Beginning of account section
                current_account = row[1]
                if current_account not in self.accounts:
                    self.accounts.append(current_account)
            elif row[0] == 'EOA':
                # End of account section
                current_account = None
            elif row[0] == 'BOS':
                # Beginning of section
                current_section = row[1]
                section_headers = []
            elif row[0] == 'EOS':
                # End of section
                current_section = None
                section_headers = []
            elif current_section:
                # We're inside a section
                # First non-BOS/EOS row after BOS is headers
                if not section_headers:
                    section_headers = row
                else:
                    # This is a data row - process based on section type
                    self._process_row(current_section, section_headers, row, current_account)

            i += 1

        # Calculate totals
        self._calculate_totals()

        return self.to_dict()

    def _process_row(self, section: str, headers: List[str], row: List[str], account: str):
        """Process a data row based on section type."""

        # Filter by account if specified
        if self.account_id and account and account != self.account_id:
            return

        # Create dict from row
        if len(row) != len(headers):
            return

        data = dict(zip(headers, row))

        if section == 'EQUT':
            # NAV data
            self._process_nav(data)
        elif section == 'CNAV':
            # Change in NAV
            self._process_nav_change(data)
        elif section == 'CTRN':
            # Cash transactions
            self._process_cash_transaction(data)
        elif section == 'POST':
            # Positions
            self._process_position(data)

    def _process_nav(self, data: Dict[str, str]):
        """Process NAV row."""
        try:
            report_date = data.get('ReportDate', '')
            if not report_date or report_date == '':
                return

            total = float(data.get('Total', 0) or 0)

            # Skip zero NAV days
            if total == 0:
                return

            nav = DailyNAV(
                date=self._format_date(report_date),
                cash=float(data.get('Cash', 0) or 0),
                stock=float(data.get('Stock', 0) or 0),
                bonds=float(data.get('Bonds', 0) or 0),
                options=float(data.get('Options', 0) or 0),
                dividends_accrual=float(data.get('DividendAccruals', 0) or 0),
                interest_accrual=float(data.get('InterestAccruals', 0) or 0),
                total=total
            )

            # Avoid duplicates
            if not any(n.date == nav.date for n in self.nav_history):
                self.nav_history.append(nav)

        except Exception as e:
            pass

    def _process_nav_change(self, data: Dict[str, str]):
        """Process Change in NAV row."""
        try:
            to_date = data.get('ToDate', '')
            if not to_date:
                return

            ending = float(data.get('EndingValue', 0) or 0)
            if ending == 0:
                return

            change = NAVChange(
                date=self._format_date(to_date),
                starting_value=float(data.get('StartingValue', 0) or 0),
                ending_value=ending,
                mtm=float(data.get('Mtm', 0) or 0),
                realized=float(data.get('Realized', 0) or 0),
                unrealized_change=float(data.get('ChangeInUnrealized', 0) or 0),
                deposits_withdrawals=float(data.get('DepositsWithdrawals', 0) or 0),
                dividends=float(data.get('Dividends', 0) or 0),
                interest=float(data.get('Interest', 0) or 0),
                commissions=float(data.get('Commissions', 0) or 0),
                fees=float(data.get('BrokerFees', 0) or 0) + float(data.get('OtherFees', 0) or 0),
                other=float(data.get('OtherIncome', 0) or 0)
            )

            if not any(n.date == change.date for n in self.nav_changes):
                self.nav_changes.append(change)

        except Exception as e:
            pass

    def _process_cash_transaction(self, data: Dict[str, str]):
        """Process cash transaction row."""
        try:
            date_str = data.get('Date/Time', '')
            if not date_str:
                return

            amount = float(data.get('Amount', 0) or 0)
            if amount == 0:
                return

            txn_type = data.get('Type', '')

            txn = CashTransaction(
                date=self._format_date(date_str.split(';')[0]),
                amount=amount,
                type=txn_type,
                description=data.get('Description', ''),
                symbol=data.get('Symbol', '')
            )

            self.cash_transactions.append(txn)

        except Exception as e:
            pass

    def _process_position(self, data: Dict[str, str]):
        """Process position row."""
        try:
            report_date = data.get('ReportDate', '')
            symbol = data.get('Symbol', '')

            if not report_date or not symbol:
                return

            quantity = float(data.get('Quantity', 0) or 0)
            if quantity == 0:
                return

            date_key = self._format_date(report_date)

            pos = Position(
                date=date_key,
                symbol=symbol,
                description=data.get('Description', ''),
                quantity=quantity,
                mark_price=float(data.get('MarkPrice', 0) or 0),
                position_value=float(data.get('PositionValue', 0) or 0),
                cost_basis=float(data.get('CostBasisMoney', 0) or 0),
                unrealized_pnl=float(data.get('FifoPnlUnrealized', 0) or 0),
                percent_of_nav=float(data.get('PercentOfNAV', 0) or 0)
            )

            if date_key not in self.positions:
                self.positions[date_key] = []

            # Avoid duplicates
            if not any(p.symbol == pos.symbol and p.date == pos.date for p in self.positions[date_key]):
                self.positions[date_key].append(pos)

        except Exception as e:
            pass

    def _format_date(self, date_str: str) -> str:
        """Convert YYYYMMDD to YYYY-MM-DD."""
        if len(date_str) >= 8:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return date_str

    def _calculate_totals(self):
        """Calculate summary totals from transactions."""
        for txn in self.cash_transactions:
            txn_type = txn.type.lower()

            if 'deposit' in txn_type or 'withdrawal' in txn_type:
                if txn.amount > 0:
                    self.total_deposits += txn.amount
                else:
                    self.total_withdrawals += abs(txn.amount)
            elif 'dividend' in txn_type:
                self.total_dividends += txn.amount
            elif 'interest' in txn_type:
                self.total_interest += txn.amount
            elif 'fee' in txn_type or 'commission' in txn_type:
                self.total_fees += abs(txn.amount)

    def get_nav_dataframe(self) -> pd.DataFrame:
        """Get NAV history as DataFrame."""
        if not self.nav_history:
            return pd.DataFrame()

        df = pd.DataFrame([asdict(n) for n in self.nav_history])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df

    def get_transactions_dataframe(self) -> pd.DataFrame:
        """Get cash transactions as DataFrame."""
        if not self.cash_transactions:
            return pd.DataFrame()

        df = pd.DataFrame([asdict(t) for t in self.cash_transactions])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df

    def calculate_performance(self) -> Dict[str, Any]:
        """Calculate various performance metrics."""

        nav_df = self.get_nav_dataframe()

        if nav_df.empty or len(nav_df) < 2:
            return {}

        # Get first and last NAV
        first_nav = nav_df.iloc[0]['total']
        last_nav = nav_df.iloc[-1]['total']
        first_date = nav_df.iloc[0]['date']
        last_date = nav_df.iloc[-1]['date']

        # Net deposits
        net_deposits = self.total_deposits - self.total_withdrawals

        # True profit (NAV - Net Deposits)
        true_profit = last_nav - net_deposits
        true_profit_pct = (true_profit / net_deposits * 100) if net_deposits > 0 else 0

        # Simple return (not accounting for deposits)
        simple_return = last_nav - first_nav
        simple_return_pct = (simple_return / first_nav * 100) if first_nav > 0 else 0

        # Time-Weighted Return (TWR) - approximation using daily returns
        nav_df['daily_return'] = nav_df['total'].pct_change()
        nav_df['cumulative_return'] = (1 + nav_df['daily_return']).cumprod() - 1
        twr = nav_df['cumulative_return'].iloc[-1] * 100 if len(nav_df) > 1 else 0

        # Period calculations
        days = (last_date - first_date).days

        # Annualized return
        if days > 0:
            annualized_return = ((last_nav / first_nav) ** (365 / days) - 1) * 100 if first_nav > 0 else 0
        else:
            annualized_return = 0

        # Calculate MTD, YTD
        today = last_date

        # MTD
        month_start = today.replace(day=1)
        mtd_start_nav = nav_df[nav_df['date'] >= month_start].iloc[0]['total'] if len(nav_df[nav_df['date'] >= month_start]) > 0 else last_nav
        mtd_return = ((last_nav - mtd_start_nav) / mtd_start_nav * 100) if mtd_start_nav > 0 else 0

        # YTD
        year_start = today.replace(month=1, day=1)
        ytd_df = nav_df[nav_df['date'] >= year_start]
        ytd_start_nav = ytd_df.iloc[0]['total'] if len(ytd_df) > 0 else last_nav
        ytd_return = ((last_nav - ytd_start_nav) / ytd_start_nav * 100) if ytd_start_nav > 0 else 0

        # Best/Worst days
        nav_df['daily_pnl'] = nav_df['total'].diff()
        best_day = nav_df.loc[nav_df['daily_pnl'].idxmax()] if not nav_df['daily_pnl'].isna().all() else None
        worst_day = nav_df.loc[nav_df['daily_pnl'].idxmin()] if not nav_df['daily_pnl'].isna().all() else None

        # Max drawdown
        nav_df['cummax'] = nav_df['total'].cummax()
        nav_df['drawdown'] = (nav_df['total'] - nav_df['cummax']) / nav_df['cummax'] * 100
        max_drawdown = nav_df['drawdown'].min()

        # Volatility (annualized)
        daily_vol = nav_df['daily_return'].std()
        annual_vol = daily_vol * np.sqrt(252) * 100 if not np.isnan(daily_vol) else 0

        # Sharpe ratio (assuming 5% risk-free rate)
        risk_free = 5.0
        sharpe = (annualized_return - risk_free) / annual_vol if annual_vol > 0 else 0

        return {
            'summary': {
                'first_date': first_date.strftime('%Y-%m-%d'),
                'last_date': last_date.strftime('%Y-%m-%d'),
                'days': days,
                'first_nav': round(first_nav, 2),
                'last_nav': round(last_nav, 2),
            },
            'deposits': {
                'total_deposits': round(self.total_deposits, 2),
                'total_withdrawals': round(self.total_withdrawals, 2),
                'net_deposits': round(net_deposits, 2),
            },
            'income': {
                'dividends': round(self.total_dividends, 2),
                'interest': round(self.total_interest, 2),
                'fees_paid': round(self.total_fees, 2),
            },
            'returns': {
                'true_profit': round(true_profit, 2),
                'true_profit_pct': round(true_profit_pct, 2),
                'simple_return': round(simple_return, 2),
                'simple_return_pct': round(simple_return_pct, 2),
                'twr': round(twr, 2),
                'annualized': round(annualized_return, 2),
                'mtd': round(mtd_return, 2),
                'ytd': round(ytd_return, 2),
            },
            'risk': {
                'max_drawdown': round(max_drawdown, 2),
                'volatility': round(annual_vol, 2),
                'sharpe_ratio': round(sharpe, 2),
            },
            'extremes': {
                'best_day': {
                    'date': best_day['date'].strftime('%Y-%m-%d') if best_day is not None else None,
                    'pnl': round(best_day['daily_pnl'], 2) if best_day is not None else None,
                },
                'worst_day': {
                    'date': worst_day['date'].strftime('%Y-%m-%d') if worst_day is not None else None,
                    'pnl': round(worst_day['daily_pnl'], 2) if worst_day is not None else None,
                },
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert parsed data to dictionary."""
        return {
            'accounts': self.accounts,
            'nav_history': [asdict(n) for n in sorted(self.nav_history, key=lambda x: x.date)],
            'nav_changes': [asdict(n) for n in sorted(self.nav_changes, key=lambda x: x.date)],
            'cash_transactions': [asdict(t) for t in sorted(self.cash_transactions, key=lambda x: x.date)],
            'positions_by_date': {k: [asdict(p) for p in v] for k, v in self.positions.items()},
            'totals': {
                'total_deposits': self.total_deposits,
                'total_withdrawals': self.total_withdrawals,
                'net_deposits': self.total_deposits - self.total_withdrawals,
                'total_dividends': self.total_dividends,
                'total_interest': self.total_interest,
                'total_fees': self.total_fees,
            },
            'performance': self.calculate_performance()
        }

    def to_json(self, filepath: str = None) -> str:
        """Export to JSON."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, default=str)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str


def parse_flexquery_csv(filepath: str, account_id: str = None) -> Dict[str, Any]:
    """Convenience function to parse FlexQuery CSV."""
    parser = IBKRFlexQueryParser(filepath, account_id)
    return parser.parse()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ibkr_flexquery_parser.py <csv_file> [account_id]")
        sys.exit(1)

    filepath = sys.argv[1]
    account_id = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Parsing {filepath}...")

    parser = IBKRFlexQueryParser(filepath, account_id)
    data = parser.parse()

    print(f"\n📊 Accounts found: {data['accounts']}")
    print(f"📈 NAV data points: {len(data['nav_history'])}")
    print(f"💰 Cash transactions: {len(data['cash_transactions'])}")

    if data['performance']:
        perf = data['performance']
        print(f"\n{'='*50}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*50}")

        print(f"\n📅 Period: {perf['summary']['first_date']} to {perf['summary']['last_date']} ({perf['summary']['days']} days)")
        print(f"📊 NAV: ${perf['summary']['first_nav']:,.0f} → ${perf['summary']['last_nav']:,.0f}")

        print(f"\n💵 Cash Flow:")
        print(f"   Deposits:     ${perf['deposits']['total_deposits']:,.0f}")
        print(f"   Withdrawals:  ${perf['deposits']['total_withdrawals']:,.0f}")
        print(f"   Net Deposits: ${perf['deposits']['net_deposits']:,.0f}")

        print(f"\n💰 Income:")
        print(f"   Dividends: ${perf['income']['dividends']:,.2f}")
        print(f"   Interest:  ${perf['income']['interest']:,.2f}")
        print(f"   Fees Paid: ${perf['income']['fees_paid']:,.2f}")

        print(f"\n📈 Returns:")
        print(f"   True Profit:    ${perf['returns']['true_profit']:+,.0f} ({perf['returns']['true_profit_pct']:+.2f}%)")
        print(f"   TWR:            {perf['returns']['twr']:+.2f}%")
        print(f"   Annualized:     {perf['returns']['annualized']:+.2f}%")
        print(f"   MTD:            {perf['returns']['mtd']:+.2f}%")
        print(f"   YTD:            {perf['returns']['ytd']:+.2f}%")

        print(f"\n⚠️ Risk:")
        print(f"   Max Drawdown:  {perf['risk']['max_drawdown']:.2f}%")
        print(f"   Volatility:    {perf['risk']['volatility']:.2f}%")
        print(f"   Sharpe Ratio:  {perf['risk']['sharpe_ratio']:.2f}")

        if perf['extremes']['best_day']['date']:
            print(f"\n🎯 Best Day:  {perf['extremes']['best_day']['date']} (+${perf['extremes']['best_day']['pnl']:,.0f})")
            print(f"   Worst Day: {perf['extremes']['worst_day']['date']} (${perf['extremes']['worst_day']['pnl']:,.0f})")

    # Save JSON
    output_path = filepath.replace('.csv', '_parsed.json')
    parser.to_json(output_path)
    print(f"\n✅ Saved to {output_path}")