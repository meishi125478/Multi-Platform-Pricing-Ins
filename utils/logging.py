"""Unified logging system for ins_pricing package.

Provides consistent logging across all modules with configurable levels
via environment variables.

Example:
    >>> from ins_pricing.utils import get_logger
    >>> logger = get_logger("ins_pricing.trainer")
    >>> logger.info("Training started")
    [INFO][ins_pricing.trainer] Training started

Environment variables:
    INS_PRICING_LOG_LEVEL: Set log level (DEBUG, INFO, WARNING, ERROR)
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=1)
def _get_package_logger() -> logging.Logger:
    """Get or create the package-level logger with consistent formatting."""
    logger = logging.getLogger("ins_pricing")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s][%(name)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        level = os.environ.get("INS_PRICING_LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
    return logger


def get_logger(name: str = "ins_pricing") -> logging.Logger:
    """Get a logger with the given name, inheriting package-level settings.

    Args:
        name: Logger name, typically module name like 'ins_pricing.trainer'

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger("ins_pricing.trainer.ft")
        >>> logger.info("Training started")
    """
    _get_package_logger()
    return logging.getLogger(name)


def configure_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """Configure package-wide logging settings.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
    """
    logger = _get_package_logger()

    if level is not None:
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(log_level)

    if format_string is not None and logger.handlers:
        formatter = logging.Formatter(format_string)
        for handler in logger.handlers:
            handler.setFormatter(formatter)
