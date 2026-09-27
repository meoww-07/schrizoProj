import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_synthetic(n=60, p_s=10, p_f=60, n_confounds=3, signal="structural", effect=1.5, seed=0):
    """Design matrix with blocks structural | functional | confounds and balanced labels."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    rng.shuffle(y)
    Xs = rng.normal(size=(n, p_s))
    Xf = rng.normal(size=(n, p_f))
    C = rng.normal(size=(n, n_confounds))
    if signal in ("structural", "both"):
        Xs[:, :3] += effect * y[:, None]
    if signal in ("functional", "both"):
        Xf[:, :5] += effect * y[:, None]
    X = np.hstack([Xs, Xf, C])
    blocks = {
        "structural": np.arange(p_s),
        "functional": np.arange(p_s, p_s + p_f),
        "confounds": np.arange(p_s + p_f, p_s + p_f + n_confounds),
    }
    return X, y, blocks


@pytest.fixture
def synthetic():
    return make_synthetic
