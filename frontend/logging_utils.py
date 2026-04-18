"""Lightweight logging helpers for frontend modules."""

from __future__ import annotations

import logging
from typing import Optional, Union


def get_frontend_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def log_print(
    logger: logging.Logger,
    *args,
    level: Optional[Union[int, str]] = None,
    **kwargs,
) -> None:
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
