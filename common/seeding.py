"""Seed every RNG a task might use, so a run can be reproduced from one int.

Frameworks that are available are seeded; frameworks that are merely
importable but not already in use by the caller are *not* imported just to
seed them (that would force a slow ``import torch``/``tensorflow`` even for
tasks, e.g. Covertype, that don't need them).
"""

import random
import sys

import numpy as np


def set_seed(seed: int) -> None:
    """Fix the seed of Python's ``random``, NumPy, and (if already imported
    by the caller) PyTorch and TensorFlow, for one reproducible run."""
    random.seed(seed)
    np.random.seed(seed)

    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if "tensorflow" in sys.modules:
        sys.modules["tensorflow"].random.set_seed(seed)
