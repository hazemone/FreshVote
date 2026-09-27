import numpy as np
import pytest

from sim_cache import CacheConfig, policy_weights, _row_quantiles, run_cache


def _one_row(ages, q=None):
    ages = np.asarray([ages], dtype=float)
    q = np.ones_like(ages) if q is None else np.asarray([q], dtype=float)
    return ages, q, np.ones_like(ages, dtype=bool)


def test_row_quantiles_match_numpy_percentile():
    rng = np.random.default_rng(0)
    ages = rng.integers(0, 200, size=(5, 12)).astype(float)
    valid = rng.random((5, 12)) < 0.7
    valid[:, 0] = True
    (q25, med, q75), n = _row_quantiles(ages, valid)
    for i in range(5):
        a = ages[i, valid[i]]
        assert med[i] == pytest.approx(np.median(a))
        assert q25[i] == pytest.approx(np.percentile(a, 25))
        assert q75[i] == pytest.approx(np.percentile(a, 75))
        assert n[i] == valid[i].sum()


def test_baseline_ignores_age_and_ttl_rejects_old_entries():
    ages, q, valid = _one_row([1, 50, 500])
    assert np.allclose(policy_weights(CacheConfig(policy="baseline"), ages, q, valid), 1.0)
    w = policy_weights(CacheConfig(policy="hard_ttl", ttl=40), ages, q, valid)
    assert w.tolist() == [[1.0, 0.0, 0.0]]


def test_fixed_decay_is_exponential_in_age():
    ages, q, valid = _one_row([0, 10])
    w = policy_weights(CacheConfig(policy="fixed_decay", lam=0.1), ages, q, valid)
    assert w[0, 1] == pytest.approx(np.exp(-1.0))


def test_freshvote_scales_with_local_median_and_rejects_outliers():
    ages, q, valid = _one_row([10, 10, 10, 10, 200])
    w = policy_weights(CacheConfig(policy="adaptive_freshvote", kappa=1.5), ages, q, valid)
    assert w[0, 0] == pytest.approx(0.5)      # half-life equals the median age
    assert w[0, 4] == 0.0                     # beyond median + kappa * IQR
    shifted = policy_weights(CacheConfig(policy="adaptive_freshvote", kappa=1.5), ages * 3, q, valid)
    assert shifted[0, 0] == pytest.approx(0.5)  # invariant to a common rescaling of ages


def test_freshvote_cold_start_uses_baseline_weights():
    ages, q, valid = _one_row([5, 400])
    w = policy_weights(CacheConfig(policy="adaptive_freshvote", cold_start=3), ages, q, valid)
    assert np.allclose(w, 1.0)


def test_invalid_entries_get_zero_weight():
    ages = np.array([[1.0, 2.0]])
    q = np.ones((1, 2))
    valid = np.array([[True, False]])
    for pol in ("baseline", "hard_ttl", "fixed_decay", "adaptive_freshvote"):
        assert policy_weights(CacheConfig(policy=pol), ages, q, valid)[0, 1] == 0.0


def test_short_run_is_deterministic_and_policies_share_channel_streams():
    kw = dict(num_agents=30, steps=300, swap_tick=150, beacon_period=5, seed=3)
    a = run_cache(CacheConfig(policy="baseline", **kw))
    b = run_cache(CacheConfig(policy="baseline", **kw))
    c = run_cache(CacheConfig(policy="adaptive_freshvote", **kw))
    assert a == b
    assert a["delivered"] == c["delivered"] and a["sent"] == c["sent"]
