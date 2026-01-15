"""File I/O and parameter loading utilities.

This module contains:
- IOUtils class for loading parameters from JSON/CSV/TSV files
- csv_to_dict() for CSV file handling
- File path sanitization utilities
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class IOUtils:
    """File and path utilities for model parameters and configs."""

    @staticmethod
    def csv_to_dict(file_path: str) -> List[Dict[str, Any]]:
        """Load CSV file as list of dictionaries.

        Args:
            file_path: Path to CSV file

        Returns:
            List of dictionaries, one per row
        """
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return [
                dict(filter(lambda item: item[0] != '', row.items()))
                for row in reader
            ]

    @staticmethod
    def ensure_parent_dir(file_path: str) -> None:
        """Create parent directories when missing.

        Args:
            file_path: Path to file whose parent directory should be created
        """
        directory = Path(file_path).parent
        if directory and not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_params_dict(params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter index-like columns such as "Unnamed: 0" from pandas I/O.

        Args:
            params: Parameter dictionary

        Returns:
            Sanitized parameter dictionary
        """
        return {
            k: v
            for k, v in (params or {}).items()
            if k and not str(k).startswith("Unnamed")
        }

    @staticmethod
    def load_params_file(path: str) -> Dict[str, Any]:
        """Load parameter dict from JSON/CSV/TSV files.

        Supported formats:
        - JSON: accept dict or {"best_params": {...}} wrapper
        - CSV/TSV: read the first row as params

        Args:
            path: Path to parameter file

        Returns:
            Parameter dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported or invalid
        """
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"params file not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".json":
            payload = json.loads(file_path.read_text(
                encoding="utf-8", errors="replace"))
            if isinstance(payload, dict) and "best_params" in payload:
                payload = payload.get("best_params") or {}
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Invalid JSON params file (expect dict): {file_path}")
            return IOUtils._sanitize_params_dict(dict(payload))

        if suffix in (".csv", ".tsv"):
            df = pd.read_csv(file_path, sep="\t" if suffix == ".tsv" else ",")
            if df.empty:
                raise ValueError(f"Empty params file: {file_path}")
            params = df.iloc[0].to_dict()
            return IOUtils._sanitize_params_dict(params)

        raise ValueError(
            f"Unsupported params file type '{suffix}': {file_path}")


# Backward compatibility function wrapper
def csv_to_dict(file_path: str) -> List[Dict[str, Any]]:
    """Load CSV file as list of dictionaries (legacy function).

    Args:
        file_path: Path to CSV file

    Returns:
        List of dictionaries, one per row
    """
    return IOUtils.csv_to_dict(file_path)


def ensure_parent_dir(file_path: str) -> None:
    """Create parent directories when missing (legacy function)."""
    IOUtils.ensure_parent_dir(file_path)
