"""
Cooperative-game analysis of a complete 2^N reward-component ablation, plus the
significance machinery that goes with it.

Why this module exists
----------------------
A leave-one-out ablation reports, for component i, the single number

    LOO(i) = v(N) - v(N \\ {i})

which answers "what does i add on top of everything else?".  Its mirror image,

    AOI(i) = v({i}) - v(empty)

answers "what does i achieve on its own?".  These two numbers routinely disagree
by an order of magnitude, and when they do, neither deserves to be called "the
effect of component i".  With the complete lattice available we can stop
choosing a context and average over all of them, which is exactly the Shapley
value:

    phi_i = sum_{S subset N\\{i}} |S|! (n-|S|-1)! / n! * [v(S + i) - v(S)]

It satisfies efficiency (sum_i phi_i = v(N) - v(empty)), symmetry, linearity and
the null-player property, and it is the unique value that does.  The Moebius
(Harsanyi dividend) transform of the same table,

    m(S) = sum_{T subset S} (-1)^{|S \\ T|} v(T),

decomposes the ablation into a main effect per component plus an explicit
interaction term for every pair, triple, ... .  A negative pairwise dividend
means the two components are SUBSTITUTES (each covers for the other, which is
precisely why leave-one-out finds nothing when you drop either alone); a
positive one means they are COMPLEMENTS.

Everything here is exact arithmetic on the lattice -- no model is fitted, no
distributional assumption is made.  Uncertainty comes from re-doing the exact
computation independently within each training seed and then treating the seeds
as the sample, which is the correct unit of analysis for a deep-RL study.
"""
from itertools import combinations
from math import factorial

import numpy as np

__all__ = [
    "subsets", "mobius_transform", "shapley_values", "interaction_index",
    "loo_effects", "aoi_effects", "lattice_table", "component_analysis",
    "sign_flip_test", "bootstrap_ci", "tost_equivalence", "cliffs_delta",
    "holm_bonferroni", "hierarchical_bootstrap_ci",
]


# --------------------------------------------------------------------------- #
#  Lattice helpers
# --------------------------------------------------------------------------- #
def subsets(n):
    """All 2^n index tuples, ordered by (size, lexicographic)."""
    out = []
    for k in range(n + 1):
        out.extend(combinations(range(n), k))
    return out


def _mask(idx, n):
    m = [False] * n
    for i in idx:
        m[i] = True
    return tuple(m)


# --------------------------------------------------------------------------- #
#  Exact cooperative-game quantities
# --------------------------------------------------------------------------- #
def mobius_transform(v, n):
    """Harsanyi dividends.  `v` maps an index tuple -> value.  Returns dict."""
    m = {}
    for S in subsets(n):
        s = 0.0
        for k in range(len(S) + 1):
            for T in combinations(S, k):
                s += ((-1.0) ** (len(S) - len(T))) * v[T]
        m[S] = s
    return m


def shapley_values(v, n):
    """Exact Shapley value of every component.

    Computed from the definition rather than from the Moebius transform so that
    the two can be cross-checked against each other (see the unit tests).
    """
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for k in range(n):
            for S in combinations(others, k):
                w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                phi[i] += w * (v[tuple(sorted(S + (i,)))] - v[tuple(sorted(S))])
    return phi


def interaction_index(v, n, order=2):
    """Shapley interaction index for every subset of size `order`.

    I(S) = sum_{T superset S} m(T) / (|T| - |S| + 1),  the natural extension of
    the Shapley value to coalitions (Grabisch & Roubens, 1999).  For |S| = 1 it
    reduces to the Shapley value, which the tests verify.
    """
    m = mobius_transform(v, n)
    out = {}
    for S in combinations(range(n), order):
        acc = 0.0
        sS = set(S)
        for T, mt in m.items():
            if sS.issubset(T):
                acc += mt / (len(T) - order + 1)
        out[S] = acc
    return out


def loo_effects(v, n):
    """v(N) - v(N \\ {i}) -- the classical leave-one-out ablation number."""
    N = tuple(range(n))
    return np.array([v[N] - v[tuple(j for j in N if j != i)] for i in range(n)])


def aoi_effects(v, n):
    """v({i}) - v(empty) -- the add-one-in (solo) number."""
    return np.array([v[(i,)] - v[()] for i in range(n)])


# --------------------------------------------------------------------------- #
#  Building the value function from experimental results
# --------------------------------------------------------------------------- #
def lattice_table(per_run, metric, name_fn, n=5, seed_col="seed",
                  config_col="config"):
    """Build {seed: {index_tuple: value}} from a per-(arm, seed) results frame.

    `name_fn(index_tuple)` must return the arm name used in `per_run`.  A seed is
    dropped (with its name recorded) if any lattice cell is missing for it, since
    the Shapley computation needs the complete table and silently imputing a
    hole would fabricate an effect.
    """
    all_S = subsets(n)
    tables, dropped = {}, []
    for sd, g in per_run.groupby(seed_col):
        lut = dict(zip(g[config_col], g[metric]))
        tab, ok = {}, True
        for S in all_S:
            arm = name_fn(S)
            if arm not in lut or not np.isfinite(lut[arm]):
                ok = False
                break
            tab[S] = float(lut[arm])
        if ok:
            tables[int(sd)] = tab
        else:
            dropped.append(int(sd))
    return tables, dropped


def component_analysis(tables, n=5, n_boot=20000, rng_seed=0):
    """Per-seed exact analysis, then seed-level inference.

    Returns a dict with, for every component: the Shapley value, the leave-one-out
    effect and the add-one-in effect -- each as a mean over seeds with a
    percentile bootstrap CI and an exact sign-flip permutation p-value -- plus the
    pairwise interaction indices and the Moebius dividends.
    """
    seeds = sorted(tables)
    if not seeds:
        raise ValueError("no complete lattice for any seed")
    phi = np.vstack([shapley_values(tables[s], n) for s in seeds])
    loo = np.vstack([loo_effects(tables[s], n) for s in seeds])
    aoi = np.vstack([aoi_effects(tables[s], n) for s in seeds])

    pairs = list(combinations(range(n), 2))
    inter = np.vstack([[interaction_index(tables[s], n, 2)[p] for p in pairs]
                       for s in seeds])

    all_S = subsets(n)
    mob = np.vstack([[mobius_transform(tables[s], n)[S] for S in all_S]
                     for s in seeds])

    total = np.array([tables[s][tuple(range(n))] - tables[s][()] for s in seeds])

    def summarise(mat):
        out = []
        for c in range(mat.shape[1]):
            v = mat[:, c]
            mu, lo, hi = bootstrap_ci(v, n_boot=n_boot, seed=rng_seed)
            out.append({"mean": mu, "lo": lo, "hi": hi,
                        "median": float(np.median(v)),
                        "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                        "p": sign_flip_test(v)})
        return out

    return {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "shapley": summarise(phi),
        "loo": summarise(loo),
        "aoi": summarise(aoi),
        "interaction": summarise(inter),
        "interaction_pairs": pairs,
        "mobius": summarise(mob),
        "mobius_sets": all_S,
        "shapley_raw": phi,
        "loo_raw": loo,
        "aoi_raw": aoi,
        "interaction_raw": inter,
        "total_mean": float(np.mean(total)),
        "total_raw": total,
        "efficiency_residual": float(np.max(np.abs(phi.sum(axis=1) - total))),
    }


# --------------------------------------------------------------------------- #
#  Seed-level inference
# --------------------------------------------------------------------------- #
def sign_flip_test(values, n_perm=100000, seed=0):
    """Exact (or Monte-Carlo) two-sided sign-flip permutation test of H0: mean=0.

    The right test for "is this component's effect distinguishable from zero?"
    when the sample is a handful of training seeds: it assumes only that the
    per-seed effects are symmetric about their centre under H0, and unlike the
    Wilcoxon signed-rank test it does not throw away magnitude information.
    With n <= 20 seeds every one of the 2^n sign assignments is enumerated, so
    the p-value is exact.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return np.nan
    if np.allclose(v, 0.0):
        return 1.0
    obs = abs(float(np.mean(v)))
    if n <= 20:                                    # exact enumeration
        signs = (((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1)
        means = np.abs(signs @ v) / n
        return float((means >= obs - 1e-15).mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    means = np.abs(signs @ v) / n
    return float((means >= obs - 1e-15).mean())


def bootstrap_ci(values, n_boot=20000, alpha=0.05, seed=0):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return (float(v.mean()), float(np.quantile(bs, alpha / 2)),
            float(np.quantile(bs, 1 - alpha / 2)))


def hierarchical_bootstrap_ci(per_episode, metric, seed_col="train_seed",
                              n_boot=4000, alpha=0.05, seed=0):
    """Resample seeds, then episodes within seed.

    Deep-RL papers routinely bootstrap episodes while holding the seed mixture
    fixed, which prices only scenario noise and treats the seeds as if they were
    the population.  Resampling at both levels prices the question a reader
    actually cares about: "if I retrained this, what would I get?"
    """
    rng = np.random.default_rng(seed)
    groups = [g[metric].to_numpy(float) for _, g in per_episode.groupby(seed_col)]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g)]
    if not groups:
        return np.nan, np.nan, np.nan
    point = float(np.mean([g.mean() for g in groups]))
    draws = np.empty(n_boot)
    k = len(groups)
    for b in range(n_boot):
        pick = rng.integers(0, k, size=k)
        draws[b] = np.mean([rng.choice(groups[i], size=len(groups[i]),
                                       replace=True).mean() for i in pick])
    return point, float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def tost_equivalence(values, margin, n_perm=100000, seed=0):
    """Two one-sided tests for H1: |mean| < margin  (i.e. PRACTICAL EQUIVALENCE).

    A non-significant difference test is not evidence of no effect -- it is the
    absence of evidence.  If the claim in the paper is "removing R4 does not
    change success", that claim has to be tested directly, and TOST is the
    standard way: we reject the null of a difference at least as large as
    `margin` in either direction.  Returns (p_equivalence, lo90, hi90); the
    equivalence claim holds at alpha when the 90% CI lies inside +/-margin.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    n = len(v)
    if n < 2:
        return np.nan, np.nan, np.nan
    mu = float(v.mean())
    se = float(np.std(v, ddof=1) / np.sqrt(n))
    from scipy import stats
    if se <= 0:
        p = 0.0 if abs(mu) < margin else 1.0
        return p, mu, mu
    t_lo = (mu + margin) / se           # H0: mu <= -margin
    t_hi = (mu - margin) / se           # H0: mu >= +margin
    p = max(float(stats.t.sf(t_lo, n - 1)), float(stats.t.cdf(t_hi, n - 1)))
    half = float(stats.t.ppf(0.95, n - 1)) * se          # 90% CI == TOST at 5%
    return p, mu - half, mu + half


def cliffs_delta(a, b):
    """Non-parametric effect size in [-1, 1]; 0 means complete overlap."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def holm_bonferroni(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj
