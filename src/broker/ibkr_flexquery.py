"""
IBKR FlexQuery Integration - Fetch Deposits/Withdrawals Automatically
======================================================================

This module fetches account cash transactions (deposits, withdrawals) from IBKR
using the FlexQuery API to calculate true profit/loss.

Setup Instructions:
1. Log into IBKR Account Management
2. Go to Reports → Flex Queries → Create New
3. Create a "Cash Transactions" query with:
   - Query Type: Activity Flex Query
   - Sections: Cash Transactions (Deposits & Withdrawals)
   - Date Period: From Account Inception (or your desired start)
4. Save and note the Query ID
5. Go to Settings → Configure Flex Web Service → Generate Token
6. Save the token in your .env file as IBKR_FLEX_TOKEN

Author: HH Research Platform
"""

import os
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json


@dataclass
class CashTransaction:
    """Represents a single cash transaction (deposit/withdrawal)."""
    date: datetime
    type: str  # 'Deposit' or 'Withdrawal'
    amount: float
    currency: str
    description: str
    account_id: str


@dataclass
class CashFlowSummary:
    """Summary of all deposits and withdrawals."""
    total_deposits: float
    total_withdrawals: float
    net_deposits: float  # deposits - withdrawals
    transaction_count: int
    first_transaction_date: Optional[datetime]
    last_transaction_date: Optional[datetime]
    transactions: List[CashTransaction]


# =============================================================================
# FLEXQUERY API
# =============================================================================

FLEX_REQUEST_URL = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.SendRequest"
FLEX_STATEMENT_URL = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService.GetStatement"


def request_flex_report(token: str, query_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Request a FlexQuery report from IBKR.

    Returns:
        Tuple of (reference_code, error_message)
    """
    params = {
        't': token,
        'q': query_id,
        'v': '3'
    }

    try:
        response = requests.get(FLEX_REQUEST_URL, params=params, timeout=30)
        response.raise_for_status()

        # Parse XML response
        root = ET.fromstring(response.text)

        # Check for errors
        status = root.find('.//Status')
        if status is not None and status.text != '0':
            error_msg = root.find('.//ErrorMessage')
            return None, error_msg.text if error_msg is not None else "Unknown error"

        # Get reference code
        ref_code = root.find('.//ReferenceCode')
        if ref_code is not None:
            return ref_code.text, None

        return None, "No reference code in response"

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"
    except ET.ParseError as e:
        return None, f"XML parse error: {str(e)}"


def fetch_flex_statement(token: str, reference_code: str, max_retries: int = 10) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch the FlexQuery statement using the reference code.
    May need to retry as IBKR generates the report asynchronously.

    Returns:
        Tuple of (xml_content, error_message)
    """
    params = {
        't': token,
        'q': reference_code,
        'v': '3'
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(FLEX_STATEMENT_URL, params=params, timeout=60)
            response.raise_for_status()

            # Check if it's still being generated
            if '<FlexStatementResponse' in response.text:
                root = ET.fromstring(response.text)
                status = root.find('.//Status')

                if status is not None:
                    if status.text == '0':
                        # Success - but might be wrapper, check for actual data
                        return response.text, None
                    elif status.text in ['1', '2']:  # Still generating
                        time.sleep(2)
                        continue
                    else:
                        error_msg = root.find('.//ErrorMessage')
                        return None, error_msg.text if error_msg is not None else f"Status: {status.text}"

            # If response contains FlexQueryResponse, it's the actual data
            if '<FlexQueryResponse' in response.text:
                return response.text, None

            time.sleep(2)

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return None, f"Request failed after {max_retries} attempts: {str(e)}"
            time.sleep(2)
        except ET.ParseError:
            # Might be the actual report (not wrapped in response)
            return response.text, None

    return None, "Timed out waiting for report"


def parse_cash_transactions(xml_content: str, account_id: Optional[str] = None) -> CashFlowSummary:
    """
    Parse cash transactions from FlexQuery XML response.

    Looks for:
    - Deposits (positive amounts with type containing 'Deposit')
    - Withdrawals (negative amounts or type containing 'Withdrawal')
    """
    transactions = []

    try:
        root = ET.fromstring(xml_content)

        # Find all CashTransaction elements
        # They can be in different paths depending on report structure
        cash_txns = root.findall('.//CashTransaction')

        if not cash_txns:
            # Try alternative paths
            cash_txns = root.findall('.//CashTransactions/CashTransaction')

        if not cash_txns:
            cash_txns = root.findall('.//FlexStatement/CashTransactions/CashTransaction')

        for txn in cash_txns:
            try:
                # Get attributes (FlexQuery uses attributes, not child elements)
                txn_type = txn.get('type', txn.get('description', ''))
                amount = float(txn.get('amount', 0))
                currency = txn.get('currency', 'USD')
                date_str = txn.get('dateTime', txn.get('reportDate', ''))
                description = txn.get('description', txn_type)
                acct = txn.get('accountId', '')

                # Filter by account if specified
                if account_id and acct and acct != account_id:
                    continue

                # Determine if deposit or withdrawal
                type_lower = txn_type.lower() if txn_type else ''
                desc_lower = description.lower() if description else ''

                is_deposit = False
                is_withdrawal = False

                # Check various indicators
                if 'deposit' in type_lower or 'deposit' in desc_lower:
                    is_deposit = True
                elif 'withdrawal' in type_lower or 'withdraw' in desc_lower:
                    is_withdrawal = True
                elif 'electronic fund transfer' in desc_lower:
                    # EFT - check amount sign
                    if amount > 0:
                        is_deposit = True
                    else:
                        is_withdrawal = True
                elif 'ach' in desc_lower or 'wire' in desc_lower:
                    if amount > 0:
                        is_deposit = True
                    else:
                        is_withdrawal = True

                # Skip if not a deposit or withdrawal
                if not is_deposit and not is_withdrawal:
                    continue

                # Parse date
                txn_date = None
                for fmt in ['%Y%m%d;%H%M%S', '%Y%m%d', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S']:
                    try:
                        txn_date = datetime.strptime(date_str.split(',')[0], fmt)
                        break
                    except:
                        continue

                if not txn_date:
                    txn_date = datetime.now()

                transactions.append(CashTransaction(
                    date=txn_date,
                    type='Deposit' if is_deposit else 'Withdrawal',
                    amount=abs(amount),
                    currency=currency,
                    description=description,
                    account_id=acct
                ))

            except Exception as e:
                print(f"Error parsing transaction: {e}")
                continue

    except ET.ParseError as e:
        print(f"XML Parse error: {e}")

    # Calculate summary
    total_deposits = sum(t.amount for t in transactions if t.type == 'Deposit')
    total_withdrawals = sum(t.amount for t in transactions if t.type == 'Withdrawal')

    # Sort by date
    transactions.sort(key=lambda x: x.date)

    return CashFlowSummary(
        total_deposits=total_deposits,
        total_withdrawals=total_withdrawals,
        net_deposits=total_deposits - total_withdrawals,
        transaction_count=len(transactions),
        first_transaction_date=transactions[0].date if transactions else None,
        last_transaction_date=transactions[-1].date if transactions else None,
        transactions=transactions
    )


def fetch_deposits_withdrawals(
    token: str,
    query_id: str,
    account_id: Optional[str] = None
) -> Tuple[Optional[CashFlowSummary], Optional[str]]:
    """
    Main function to fetch and parse deposits/withdrawals from IBKR.

    Args:
        token: FlexQuery API token
        query_id: FlexQuery ID for cash transactions report
        account_id: Optional account ID to filter transactions

    Returns:
        Tuple of (CashFlowSummary, error_message)
    """
    # Step 1: Request the report
    ref_code, error = request_flex_report(token, query_id)
    if error:
        return None, f"Failed to request report: {error}"

    # Step 2: Fetch the statement
    xml_content, error = fetch_flex_statement(token, ref_code)
    if error:
        return None, f"Failed to fetch statement: {error}"

    # Step 3: Parse transactions
    try:
        summary = parse_cash_transactions(xml_content, account_id)
        return summary, None
    except Exception as e:
        return None, f"Failed to parse transactions: {str(e)}"


# =============================================================================
# CACHING & STORAGE
# =============================================================================

def get_cache_file_path(account_id: str) -> str:
    """Get path to cached cash flow data."""
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'cash_flow_{account_id}.json')


def load_cached_cash_flow(account_id: str, max_age_hours: int = 24) -> Optional[CashFlowSummary]:
    """
    Load cached cash flow data if it exists and is fresh enough.
    """
    cache_file = get_cache_file_path(account_id)

    try:
        if not os.path.exists(cache_file):
            return None

        with open(cache_file, 'r') as f:
            data = json.load(f)

        # Check age
        cached_at = datetime.fromisoformat(data.get('cached_at', '2000-01-01'))
        if datetime.now() - cached_at > timedelta(hours=max_age_hours):
            return None

        # Reconstruct summary
        transactions = [
            CashTransaction(
                date=datetime.fromisoformat(t['date']),
                type=t['type'],
                amount=t['amount'],
                currency=t['currency'],
                description=t['description'],
                account_id=t['account_id']
            )
            for t in data.get('transactions', [])
        ]

        return CashFlowSummary(
            total_deposits=data['total_deposits'],
            total_withdrawals=data['total_withdrawals'],
            net_deposits=data['net_deposits'],
            transaction_count=data['transaction_count'],
            first_transaction_date=datetime.fromisoformat(data['first_transaction_date']) if data.get('first_transaction_date') else None,
            last_transaction_date=datetime.fromisoformat(data['last_transaction_date']) if data.get('last_transaction_date') else None,
            transactions=transactions
        )

    except Exception as e:
        print(f"Error loading cache: {e}")
        return None


def save_cash_flow_cache(account_id: str, summary: CashFlowSummary):
    """Save cash flow data to cache."""
    cache_file = get_cache_file_path(account_id)

    try:
        data = {
            'cached_at': datetime.now().isoformat(),
            'total_deposits': summary.total_deposits,
            'total_withdrawals': summary.total_withdrawals,
            'net_deposits': summary.net_deposits,
            'transaction_count': summary.transaction_count,
            'first_transaction_date': summary.first_transaction_date.isoformat() if summary.first_transaction_date else None,
            'last_transaction_date': summary.last_transaction_date.isoformat() if summary.last_transaction_date else None,
            'transactions': [
                {
                    'date': t.date.isoformat(),
                    'type': t.type,
                    'amount': t.amount,
                    'currency': t.currency,
                    'description': t.description,
                    'account_id': t.account_id
                }
                for t in summary.transactions
            ]
        }

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print(f"Error saving cache: {e}")


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

def get_account_cash_flow(
    account_id: str,
    flex_token: Optional[str] = None,
    flex_query_id: Optional[str] = None,
    use_cache: bool = True,
    cache_max_age_hours: int = 24
) -> Tuple[Optional[CashFlowSummary], Optional[str]]:
    """
    Get account cash flow (deposits/withdrawals) with caching.

    First checks cache, then fetches from IBKR if needed.

    Args:
        account_id: IBKR account ID
        flex_token: FlexQuery API token (or from IBKR_FLEX_TOKEN env var)
        flex_query_id: FlexQuery ID (or from IBKR_CASH_FLOW_QUERY_ID env var)
        use_cache: Whether to use cached data
        cache_max_age_hours: Maximum age of cache in hours

    Returns:
        Tuple of (CashFlowSummary, error_message)
    """
    # Try cache first
    if use_cache:
        cached = load_cached_cash_flow(account_id, cache_max_age_hours)
        if cached:
            return cached, None

    # Get credentials from env if not provided
    token = flex_token or os.getenv('IBKR_FLEX_TOKEN')
    query_id = flex_query_id or os.getenv('IBKR_CASH_FLOW_QUERY_ID')

    if not token or not query_id:
        return None, "FlexQuery not configured. Set IBKR_FLEX_TOKEN and IBKR_CASH_FLOW_QUERY_ID in .env"

    # Fetch from IBKR
    summary, error = fetch_deposits_withdrawals(token, query_id, account_id)

    if summary:
        # Cache the results
        save_cash_flow_cache(account_id, summary)
        return summary, None

    return None, error


def calculate_true_profit(nav: float, cash_flow: CashFlowSummary) -> Dict[str, float]:
    """
    Calculate true profit including all gains (dividends, interest, capital gains).

    True Profit = Current NAV - Net Deposits
    """
    true_profit = nav - cash_flow.net_deposits
    true_profit_pct = (true_profit / cash_flow.net_deposits * 100) if cash_flow.net_deposits > 0 else 0

    return {
        'true_profit': true_profit,
        'true_profit_pct': true_profit_pct,
        'nav': nav,
        'net_deposits': cash_flow.net_deposits,
        'total_deposits': cash_flow.total_deposits,
        'total_withdrawals': cash_flow.total_withdrawals
    }


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description='Fetch IBKR deposits/withdrawals')
    parser.add_argument('--account', '-a', help='Account ID')
    parser.add_argument('--token', '-t', help='FlexQuery token (or set IBKR_FLEX_TOKEN)')
    parser.add_argument('--query-id', '-q', help='FlexQuery ID (or set IBKR_CASH_FLOW_QUERY_ID)')
    parser.add_argument('--no-cache', action='store_true', help='Skip cache')

    args = parser.parse_args()

    summary, error = get_account_cash_flow(
        account_id=args.account or '',
        flex_token=args.token,
        flex_query_id=args.query_id,
        use_cache=not args.no_cache
    )

    if error:
        print(f"Error: {error}")
    else:
        print(f"\n💰 Cash Flow Summary")
        print(f"{'='*40}")
        print(f"Total Deposits:     ${summary.total_deposits:,.2f}")
        print(f"Total Withdrawals:  ${summary.total_withdrawals:,.2f}")
        print(f"Net Deposits:       ${summary.net_deposits:,.2f}")
        print(f"Transactions:       {summary.transaction_count}")
        if summary.first_transaction_date:
            print(f"First Transaction:  {summary.first_transaction_date.strftime('%Y-%m-%d')}")
        if summary.last_transaction_date:
            print(f"Last Transaction:   {summary.last_transaction_date.strftime('%Y-%m-%d')}")

        print(f"\n📋 Transaction History")
        print(f"{'='*40}")
        for t in summary.transactions[-10:]:  # Last 10
            sign = '+' if t.type == 'Deposit' else '-'
            print(f"{t.date.strftime('%Y-%m-%d')} | {sign}${t.amount:,.2f} | {t.description[:30]}")