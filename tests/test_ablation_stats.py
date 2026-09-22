"""Checks the attribution code against games whose answers are known exactly."""
import numpy as np
import pytest

import ablation_stats as AS

N = 5


@pytest.fixture
def random_game():
    rng = np.random.default_rng(0)
    return {S: float(rng.normal()) for S in AS.subsets(N)}


def test_shapley_matches_moebius(random_game):
    phi = AS.shapley_values(random_game, N)
    m = AS.mobius_transform(random_game, N)
    phi_m = [sum(m[S] / len(S) for S in m if i in S and S) for i in range(N)]
    assert np.allclose(phi, phi_m, atol=1e-10)


def test_efficiency(random_game):
    phi = AS.shapley_values(random_game, N)
    full = random_game[tuple(range(N))] - random_game[()]
    assert abs(phi.sum() - full) < 1e-10


def test_moebius_inversion(random_game):
    m = AS.mobius_transform(random_game, N)
    for S in AS.subsets(N):
        rebuilt = sum(m[T] for T in AS.subsets(N) if set(T) <= set(S))
        assert abs(rebuilt - random_game[S]) < 1e-9


def test_additive_game_has_no_interactions():
    w = np.random.default_rng(1).normal(size=N)
    v = {S: float(sum(w[i] for i in S)) for S in AS.subsets(N)}
    assert np.allclose(AS.shapley_values(v, N), w)
    assert np.allclose(AS.loo_effects(v, N), w)
    assert np.allclose(AS.aoi_effects(v, N), w)
    assert max(abs(x) for x in AS.interaction_index(v, N, 2).values()) < 1e-10


def test_substitutes_fool_leave_one_out():
    # value 1 if either component 2 or 3 is present: LOO sees nothing,
    # Shapley splits the credit, and the interaction is negative.
    v = {S: float(2 in S or 3 in S) for S in AS.subsets(N)}
    assert AS.loo_effects(v, N)[2] == 0.0 and AS.loo_effects(v, N)[3] == 0.0
    assert AS.shapley_values(v, N)[2] == pytest.approx(0.5)
    assert AS.interaction_index(v, N, 2)[(2, 3)] < -0.9


def test_complements_have_positive_interaction():
    v = {S: float(2 in S and 3 in S) for S in AS.subsets(N)}
    assert AS.interaction_index(v, N, 2)[(2, 3)] > 0.9


def test_sign_flip_is_exact():
    assert AS.sign_flip_test([0.1] * 6) == pytest.approx(2 / 64)
    assert AS.sign_flip_test([0.1] * 5) == pytest.approx(2 / 32)


def test_tost_both_directions():
    assert AS.tost_equivalence([0.001, -0.002, 0.003, 0.0, 0.002, -0.001], 0.02)[0] < 0.05
    assert AS.tost_equivalence([0.30, 0.28, 0.35, 0.31, 0.29, 0.33], 0.02)[0] > 0.5
