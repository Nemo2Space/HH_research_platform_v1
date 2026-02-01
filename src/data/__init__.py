"""Alpha Platform - Data Ingestion Package"""
from .market import MarketDataIngester, run_ingestion
from .fundamentals import FundamentalDataIngester
from .analyst import AnalystDataIngester
from .news import NewsCollector, NewsConfig, get_source_credibility
from .sec_filings import SECFilingFetcher, SECFiling
from .sec_rag import SECChunker, SECRetriever, SECChunk
from .insider import InsiderDataFetcher, InsiderTransaction, get_institutional_holdings

__all__ = [
    "MarketDataIngester", "run_ingestion",
    "FundamentalDataIngester", "AnalystDataIngester",
    "NewsCollector", "NewsConfig", "get_source_credibility",
    "SECFilingFetcher", "SECFiling",
    "SECChunker", "SECRetriever", "SECChunk",
    "InsiderDataFetcher", "InsiderTransaction", "get_institutional_holdings"
]