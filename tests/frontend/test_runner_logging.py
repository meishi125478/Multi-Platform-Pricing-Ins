from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from ins_pricing.frontend.logging_utils import get_frontend_logger, log_print
from ins_pricing.frontend.runner import TaskRunner


def _emit_logs(tag: str, count: int = 8) -> None:
    logger = get_frontend_logger("ins_pricing.frontend.tests.runner")
    for i in range(count):
        log_print(logger, f"{tag}:{i}")
        time.sleep(0.005)


def _collect_logs(runner: TaskRunner, tag: str) -> str:
    return "\n".join(runner.run_callable(_emit_logs, tag))


def test_run_callable_isolates_logs_between_parallel_tasks() -> None:
    runner_a = TaskRunner()
    runner_b = TaskRunner()
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(_collect_logs, runner_a, "A")
        future_b = pool.submit(_collect_logs, runner_b, "B")
        logs_a = future_a.result(timeout=20)
        logs_b = future_b.result(timeout=20)

    assert "A:0" in logs_a
    assert "A:7" in logs_a
    assert "B:0" not in logs_a

    assert "B:0" in logs_b
    assert "B:7" in logs_b
    assert "A:0" not in logs_b
