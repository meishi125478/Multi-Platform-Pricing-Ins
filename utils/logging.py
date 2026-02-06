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
from typing import Optional, Union

_DEFAULT_HANDLER_FLAG = "_ins_pricing_default_handler"


@lru_cache(maxsize=1)
def _get_package_logger() -> logging.Logger:
    """Get the package-level logger."""
    return logging.getLogger("ins_pricing")


def _sync_package_logger(logger: logging.Logger) -> None:
    """Sync package logger handlers with root logger configuration.

    Behavior:
    - If root logger is configured, remove the package default handler so logs
      flow to root only (single timestamped line).
    - If root logger is not configured, attach package default handler as
      fallback so logs remain visible.
    """
    level = os.environ.get("INS_PRICING_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    root_has_handlers = bool(logging.getLogger().handlers)

    if root_has_handlers:
        for handler in list(logger.handlers):
            if getattr(handler, _DEFAULT_HANDLER_FLAG, False):
                logger.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass
        logger.propagate = True
        return

    has_default_handler = any(
        getattr(handler, _DEFAULT_HANDLER_FLAG, False) for handler in logger.handlers
    )
    if not has_default_handler:
        handler = logging.StreamHandler()
        setattr(handler, _DEFAULT_HANDLER_FLAG, True)
        formatter = logging.Formatter("[%(levelname)s][%(name)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False


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
    _sync_package_logger(_get_package_logger())
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
    _sync_package_logger(logger)

    if level is not None:
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(log_level)

    if format_string is not None and logger.handlers:
        formatter = logging.Formatter(format_string)
        for handler in logger.handlers:
            handler.setFormatter(formatter)


def log_print(
    logger: logging.Logger,
    *args,
    level: Optional[Union[int, str]] = None,
    **kwargs,
) -> None:
    """Print-like helper that routes messages to a logger.

    This preserves basic print semantics (sep/end) while ignoring file/flush,
    and it auto-detects severity when level is not provided.
    """
    sep = kwargs.get("sep", " ")
    msg = sep.join(str(arg) for arg in args)
    if not msg:
        return

    if level is None:
        lowered = msg.lstrip().lower()
        if lowered.startswith(("warn", "[warn]", "warning")):
            level_value = logging.WARNING
        elif lowered.startswith(("error", "[error]", "err")):
            level_value = logging.ERROR
        else:
            level_value = logging.INFO
    else:
        if isinstance(level, str):
            level_value = getattr(logging, level.upper(), logging.INFO)
        else:
            level_value = int(level)

    logger.log(level_value, msg)
