"""Timer that separates data loading, training and evaluation, so each
experiment reports where its time actually went (the project's protocol
always excludes control-evaluation time from the reported training time).
"""

import time
from collections import defaultdict
from contextlib import contextmanager


class PhaseTimer:
    """Accumulates elapsed seconds per named phase.

    Usage:
        timer = PhaseTimer()
        with timer.phase("load"):
            x_train, y_train = load_data()
        with timer.phase("train"):
            train(model)
        with timer.phase("eval"):
            score = evaluate(model)
        timer.times["train"]  # seconds spent training
    """

    def __init__(self) -> None:
        self.times: dict[str, float] = defaultdict(float)

    @contextmanager
    def phase(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] += time.perf_counter() - start

    def total(self, *names: str) -> float:
        """Sum of the given phases' durations, or of all phases if none are
        given."""
        selected = names or tuple(self.times)
        return sum(self.times[name] for name in selected)
