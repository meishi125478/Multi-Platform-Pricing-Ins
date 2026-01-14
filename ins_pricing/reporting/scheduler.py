from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Optional


def _next_run(run_time: str, now: Optional[datetime] = None) -> datetime:
    if now is None:
        now = datetime.now()
    hour, minute = [int(x) for x in run_time.split(":")]
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate = candidate + timedelta(days=1)
    return candidate


def schedule_daily(
    job_fn: Callable[[], None],
    *,
    run_time: str = "01:00",
    stop_event: Optional[threading.Event] = None,
) -> threading.Thread:
    """Run job_fn daily at local time HH:MM in a background thread."""
    if stop_event is None:
        stop_event = threading.Event()

    def _loop():
        while not stop_event.is_set():
            next_time = _next_run(run_time)
            sleep_seconds = (next_time - datetime.now()).total_seconds()
            if sleep_seconds > 0:
                stop_event.wait(timeout=sleep_seconds)
            if stop_event.is_set():
                break
            try:
                job_fn()
            except Exception:
                pass
            time.sleep(1)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread
