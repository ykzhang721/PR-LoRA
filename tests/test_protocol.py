import random

import numpy as np
import pytest

from pr_lora.data import split_indices
from pr_lora.statistics import holm_adjust


@pytest.mark.parametrize("total", [9984, 9867, 998, 953, 1000, 351, 410])
def test_split_matches_original_experiment(total):
    shuffled = list(range(total))
    random.Random(42).shuffle(shuffled)
    nd, ne = min(500, total * 3 // 10), min(300, total * 3 // 10)
    pool = shuffled[nd + ne :]
    random.Random(123).shuffle(pool)
    actual = split_indices(total, train_seed=123)
    assert actual == {
        "direction": shuffled[:nd],
        "eval": shuffled[nd : nd + ne],
        "train": pool[:50],
    }
    assert not set(actual["train"]) & set(actual["direction"] + actual["eval"])
    other = split_indices(total, train_seed=456)
    assert actual["direction"] == other["direction"] and actual["eval"] == other["eval"]
    assert actual["train"] != other["train"]


def test_holm_monotonicity_in_original_order():
    np.testing.assert_allclose(holm_adjust([0.04, 0.001, 0.02, 0.03]), [0.06, 0.004, 0.06, 0.06])


def test_rejects_insufficient_data():
    with pytest.raises(ValueError, match="Cannot allocate"):
        split_indices(10)
