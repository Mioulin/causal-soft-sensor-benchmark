import copy

import numpy as np

from benchmark import HORIZONS, WINDOW, fuse_weights, metrics, replay, simulate, windows


def test_prefix_invariance():
    original = simulate(1, length=80)
    changed = copy.deepcopy(original)
    for key in ["u", "obs", "proxy", "mask"]:
        changed[key][41:] = 0
    a = replay(original)
    b = replay(changed)
    for aa, bb in zip(a, b):
        np.testing.assert_array_equal(aa[:41], bb[:41])


def test_targets_and_run_boundaries():
    r = simulate(1, length=50)
    d = windows([r, simulate(2, length=50)])
    assert len(d["features"]) == 2 * (50 - max(HORIZONS) - WINDOW + 1)
    np.testing.assert_array_equal(d["targets"][0], r["obs"][WINDOW - 1 + np.array(HORIZONS)])
    assert d["mask"].sum() == len(d["features"]) * 4


def test_correlated_error_weight():
    p = np.ones((10, 4, 6))
    n = 2 * p
    m = np.ones_like(p)
    y = np.zeros_like(p)
    np.testing.assert_array_equal(fuse_weights(p, n, y, m), np.zeros(4))
    assert np.all(fuse_weights(p, n, y, m, False) > 0)


def test_observed_metric_ignores_unobserved_targets():
    d = windows([simulate(1, length=50)])
    p = d["targets"].copy()
    p[d["mask"] == 0] += 100
    assert all(r["observed_rmse"] == 0 for r in metrics(p, d))
