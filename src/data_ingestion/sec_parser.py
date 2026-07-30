"""SEC Edgar parser for supply chain relationships."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests


class SECParser:
    """Parser for SEC filings."""

    def __init__(
        self, output_dir: Path, ticker_to_cik: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the SEC Parser.

        Args:
            output_dir (Path): Directory to save outputs.
            ticker_to_cik (Optional[Dict[str, str]]): Mapping from ticker to CIK.
        """
        self.output_dir = output_dir
        self.ticker_to_cik = ticker_to_cik or {}
        self.logger = logging.getLogger(__name__)

    def download_10k_metadata(self, ticker: str) -> Dict[str, str]:
        """
        Download 10-K metadata from SEC EDGAR.

        Args:
            ticker (str): Ticker symbol.

        Returns:
            Dict[str, str]: Metadata including accession_number, filing_date, URL.
        """
        cik = self.ticker_to_cik.get(ticker, ticker)
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count=40"

        headers = {"User-Agent": "QuantEngineBot admin@quantengine.local"}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            # Placeholder for HTML parsing logic
            return {
                "accession_number": "0000000000-00-000000",
                "filing_date": "2025-01-01",
                "primaryDocument": url,
            }
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch 10-K metadata for {ticker}: {e}")
            return {}

    def extract_supply_chain_mentions(self, text: str) -> List[Dict[str, str]]:
        """
        Extract supply chain mentions from text.
        TODO: Replace with LLM extraction in Phase 2.

        Args:
            text (str): Document text.

        Returns:
            List[Dict[str, str]]: List of extracted relationships.
        """
        return []

    def build_supply_chain_graph(self) -> pd.DataFrame:
        """
        Build supply chain graph from static CSV.

        Returns:
            pd.DataFrame: Graph edges.
        """
        static_file = Path("data/external/supply_chain_manual.csv")
        out_file = self.output_dir / "supply_chain_edges.parquet"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if static_file.exists():
            df = pd.read_csv(static_file)
            self.logger.info(f"Loaded {len(df)} edges from manual CSV.")
        else:
            self.logger.warning(
                "No manual supply chain mapping found. Creating empty DataFrame."
            )
            df = pd.DataFrame(
                columns=[
                    "source_ticker",
                    "target_ticker",
                    "relationship_type",
                    "weight",
                    "data_source",
                ]
            )

        df.to_parquet(out_file, index=False)
        self.logger.info(f"Saved supply chain edges to {out_file}")

        return df
