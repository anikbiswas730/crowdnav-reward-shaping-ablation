"""Publication-quality figures for the reward-ablation study."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})

PALETTE = ["#1b6ca8", "#d1495b", "#2a9d8f", "#e9c46a", "#8e7dbe",
           "#f4845f", "#5c8001", "#7d7d7d", "#00798c", "#b26e63"]


# --------------------------------------------------------------------------- #
def plot_trajectory(env, ax=None, title="", stride=4):
    """Draw one recorded episode (env.cfg.record_trajectory must be True)."""
    traj = env.trajectory
    assert traj, "No trajectory recorded -- set record_trajectory=True in EnvConfig."
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.2))

    rob = np.array([t["robot"] for t in traj])
    hum = np.array([t["humans"] for t in traj])       # (T, N, 2)
    hr = traj[0]["hr"]
    goal = traj[0]["goal"]
    T = len(traj)

    for i in range(hum.shape[1]):
        ax.plot(hum[:, i, 0], hum[:, i, 1], color="0.72", lw=0.9, zorder=1)
    ax.plot(rob[:, 0], rob[:, 1], color=PALETTE[1], lw=2.0, zorder=3)

    for k in range(0, T, stride):
        a = 0.12 + 0.6 * k / max(T - 1, 1)
        for i in range(hum.shape[1]):
            ax.add_patch(Circle(hum[k, i], hr[i], fc="0.55", ec="none",
                                alpha=a * 0.5, zorder=2))
        ax.add_patch(Circle(rob[k], env.robot_radius, fc=PALETTE[1], ec="none",
                            alpha=a, zorder=4))

    ax.add_patch(Circle(goal, env.cfg.goal_tolerance, fc="none",
                        ec=PALETTE[2], lw=1.8, ls="--", zorder=5))
    ax.plot(*rob[0], marker="s", ms=5, color=PALETTE[1], zorder=6)
    lim = env.cfg.circle_radius + 1.4
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    return ax


# --------------------------------------------------------------------------- #
def _load_curve(run_dir, metric):
    """Prefer the per-rollout log (already windowed, small); fall back to episodes."""
    f = os.path.join(run_dir, "train_rollouts.csv")
    if os.path.exists(f):
        df = pd.read_csv(f)
        if metric in df and not df.empty:
            return df["timesteps"].values, df[metric].values
    f = os.path.join(run_dir, "train_episodes.csv")
    if os.path.exists(f):
        df = pd.read_csv(f)
        if metric in df and not df.empty:
            return df["timesteps"].values, df[metric].rolling(200, min_periods=1).mean().values
    return None, None


def plot_learning_curves(runs_root, configs, metric="success",
                         bins=50, ax=None, title=None):
    """Mean +/- s.e.m. across seeds of a training metric vs. environment steps."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 3.4))
    for ci, cfg in enumerate(configs):
        curves, grid = [], None
        for d in sorted(os.listdir(runs_root)):
            if not d.startswith(cfg + "__seed"):
                continue
            x, y = _load_curve(os.path.join(runs_root, d), metric)
            if x is None or len(x) < 2:
                continue
            grid = np.linspace(0, float(np.max(x)), bins)
            curves.append(np.interp(grid, x, y))
        if not curves:
            continue
        c = np.vstack(curves)
        mu = c.mean(0)
        se = c.std(0, ddof=1) / np.sqrt(len(c)) if len(c) > 1 else np.zeros_like(mu)
        col = PALETTE[ci % len(PALETTE)]
        ax.plot(grid, mu, color=col, lw=1.6, label=cfg)
        ax.fill_between(grid, mu - se, mu + se, color=col, alpha=0.18)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title or f"Training {metric.replace('_',' ')}")
    ax.legend(fontsize=7, ncol=2)
    return ax


def plot_seed_traces(runs_root, configs, seeds, metric="success", ncols=4):
    """One panel per arm, one thin line per seed, plus the policy std on a twin
    axis.  This is the figure that distinguishes 'never learned' (flat line)
    from 'learned then collapsed' (rises then falls) from 'exploration runaway'
    (std climbing monotonically)."""
    n = len(configs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 2.9 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for k, cfg in enumerate(configs):
        ax = axes[k]
        ax2 = ax.twinx(); ax2.grid(False)
        for si, sd in enumerate(seeds):
            rd = os.path.join(runs_root, f"{cfg}__seed{sd}")
            x, y = _load_curve(rd, metric)
            if x is None:
                continue
            ax.plot(x, y, lw=1.0, alpha=0.85, color=PALETTE[si % len(PALETTE)])
            xs, ys = _load_curve(rd, "policy_std")
            if xs is not None:
                ax2.plot(xs, ys, lw=0.9, ls=":", color="0.35", alpha=0.7)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(cfg, fontsize=8.5)
        ax.set_xlabel("steps", fontsize=7.5)
        ax.set_ylabel(metric, fontsize=7.5)
        ax2.set_ylabel("policy std (dotted)", fontsize=7, color="0.35")
        ax2.tick_params(labelsize=6.5, colors="0.35")
        ax.tick_params(labelsize=6.5)
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
def plot_ablation_bars(agg, metrics, titles=None, reference=None, ncols=3):
    """Grouped bar chart with 95% CI whiskers, one panel per metric."""
    n = len(metrics)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    order = list(agg["config"])
    for k, m in enumerate(metrics):
        ax = axes[k]
        mu = agg[f"{m}_mean"].values
        ci = agg[f"{m}_ci95"].fillna(0).values
        cols = [PALETTE[0] if c != reference else PALETTE[1] for c in order]
        ax.bar(range(len(order)), mu, yerr=ci, capsize=3,
               color=cols, edgecolor="white", linewidth=0.6)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=55, ha="right", fontsize=7)
        ax.set_title((titles or {}).get(m, m.replace("_", " ")), fontsize=9)
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
def plot_pareto(agg, x="nav_time_succ", y="collision", label_col="config",
                ax=None, annotate=True):
    """Safety / efficiency trade-off scatter."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 3.8))
    for i, r in agg.iterrows():
        ax.errorbar(r[f"{x}_mean"], r[f"{y}_mean"],
                    xerr=r.get(f"{x}_ci95", 0), yerr=r.get(f"{y}_ci95", 0),
                    fmt="o", ms=7, color=PALETTE[i % len(PALETTE)],
                    ecolor="0.6", elinewidth=0.9, capsize=2)
        if annotate:
            ax.annotate(r[label_col], (r[f"{x}_mean"], r[f"{y}_mean"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=6.5)
    ax.set_xlabel(x.replace("_", " ") + " [s]")
    ax.set_ylabel(y.replace("_", " "))
    ax.set_title("Safety / efficiency trade-off")
    return ax


# --------------------------------------------------------------------------- #
def to_latex(agg, metrics, fmt="{:.3f}", caption="", label="tab:ablation"):
    hdr = " & ".join(["Configuration"] + [m.replace("_", " ") for m in metrics])
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{l" + "c" * len(metrics) + "}", r"\toprule",
             hdr + r" \\", r"\midrule"]
    for _, r in agg.iterrows():
        cells = [str(r["config"]).replace("_", r"\_")]
        for m in metrics:
            mu, ci = r[f"{m}_mean"], r[f"{m}_ci95"]
            cells.append(f"${fmt.format(mu)}\\pm{fmt.format(0 if pd.isna(ci) else ci)}$")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)

# --------------------------------------------------------------------------- #
#  Cooperative-game figures (new in V5)
# --------------------------------------------------------------------------- #
def plot_attribution(analysis, comp_labels, metric_label="success rate",
                     ax=None, title=None):
    """Shapley value vs leave-one-out vs add-one-in, side by side per component.

    The whole point of the figure is the DISAGREEMENT between the three bars.
    Where they agree, the component's effect is context-free and leave-one-out
    was telling the truth.  Where the Shapley bar sits far from the LOO bar, the
    single-context ablation that the literature reports is an artefact of the
    context it was measured in.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.4, 3.6))
    n = len(comp_labels)
    x = np.arange(n)
    w = 0.27
    series = [("Shapley  $\\phi_i$", analysis["shapley"], PALETTE[0]),
              ("leave-one-out", analysis["loo"], PALETTE[1]),
              ("add-one-in", analysis["aoi"], PALETTE[2])]
    for k, (lbl, rows, col) in enumerate(series):
        mu = np.array([r["mean"] for r in rows])
        lo = np.array([r["lo"] for r in rows])
        hi = np.array([r["hi"] for r in rows])
        ax.bar(x + (k - 1) * w, mu, width=w, color=col, label=lbl,
               edgecolor="white", linewidth=0.6)
        ax.errorbar(x + (k - 1) * w, mu, yerr=[mu - lo, hi - mu], fmt="none",
                    ecolor="0.25", elinewidth=0.9, capsize=2.5)
    ax.axhline(0, color="0.3", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels, fontsize=8)
    ax.set_ylabel(f"$\\Delta$ {metric_label}")
    ax.set_title(title or "Component attribution: averaged over all contexts "
                          "vs. measured in one")
    ax.legend(fontsize=7.5, ncol=3)
    return ax


def plot_interaction_heatmap(analysis, comp_labels, ax=None, title=None,
                             annotate=True):
    """Pairwise Shapley interaction indices.

    Negative (blue) = SUBSTITUTES: the pair covers the same ground, so dropping
    either one alone costs nothing and leave-one-out reports a false null.
    Positive (red) = COMPLEMENTS: the pair is worth more together than the sum
    of its parts.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.0))
    n = len(comp_labels)
    M = np.full((n, n), np.nan)
    for (i, j), row in zip(analysis["interaction_pairs"], analysis["interaction"]):
        M[i, j] = M[j, i] = row["mean"]
    vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    vmax = max(vmax, 1e-9)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(comp_labels, rotation=45,
                                                ha="right", fontsize=7.5)
    ax.set_yticks(range(n)); ax.set_yticklabels(comp_labels, fontsize=7.5)
    ax.grid(False)
    if annotate:
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=9, color="0.5")
                elif np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if abs(M[i, j]) > 0.55 * vmax else "0.15")
    ax.set_title(title or "Pairwise interaction  $I(i,j)$\n"
                          "blue = substitutes, red = complements", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_lattice_landscape(tables, name_fn, metric_label="success rate",
                           ax=None, comp_labels=None):
    """Every cell of the 2^5 lattice, arranged by the number of active
    components.  It shows at a glance whether performance is a function of HOW
    MANY terms are on (it is not) or of WHICH ones (it is)."""
    from itertools import combinations
    if ax is None:
        _, ax = plt.subplots(figsize=(7.6, 3.8))
    seeds = sorted(tables)
    all_S = sorted(tables[seeds[0]].keys(), key=lambda S: (len(S), S))
    xs, ys, lbls = [], [], []
    for S in all_S:
        vals = [tables[s][S] for s in seeds]
        xs.append(len(S)); ys.append(np.mean(vals)); lbls.append(S)
        ax.scatter([len(S)] * len(vals), vals, s=9, color="0.75", zorder=1)
    ax.scatter(xs, ys, s=52, color=PALETTE[1], zorder=3, edgecolor="white",
               linewidth=0.7)
    for x, y, S in zip(xs, ys, lbls):
        tag = "".join(str(i + 1) for i in S) or "none"
        ax.annotate(tag, (x, y), textcoords="offset points", xytext=(7, -2),
                    fontsize=6.0, color="0.25")
    ax.set_xlabel("number of active reward components")
    ax.set_ylabel(metric_label)
    ax.set_xticks(range(0, 6))
    ax.set_title("The whole $2^5$ lattice: which terms are on beats how many")
    return ax


def plot_dose_response(df, x="w_jerk", y="success", ax=None, title=None,
                       logx=True, ylabel=None):
    """Mean +/- s.e.m. across seeds against a swept weight, on a log axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.4))
    g = df.groupby(x)[y].agg(["mean", "std", "count"]).reset_index()
    se = g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))
    xv = g[x].values.astype(float)
    if logx:
        xv = np.where(xv <= 0, np.nanmin(xv[xv > 0]) / 3.0 if (xv > 0).any() else 1e-4, xv)
        ax.set_xscale("log")
    ax.errorbar(xv, g["mean"], yerr=se, marker="o", ms=5, lw=1.6,
                color=PALETTE[0], ecolor="0.5", capsize=3)
    if logx and (g[x].values <= 0).any():
        ax.axvline(xv[g[x].values <= 0][0], color=PALETTE[1], ls=":", lw=1.2)
        ax.annotate("w = 0", (xv[g[x].values <= 0][0], ax.get_ylim()[0]),
                    fontsize=7, color=PALETTE[1], rotation=90,
                    textcoords="offset points", xytext=(4, 6))
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(ylabel or y.replace("_", " "))
    ax.set_title(title or f"Dose–response: {y.replace('_',' ')} vs {x}")
    return ax


def plot_density_effects(df, arms, metric="success", ax=None, title=None):
    """Same contrast measured at two crowd densities, with seed scatter."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 3.6))
    dens = sorted(df["n_humans"].unique())
    x = np.arange(len(arms))
    w = 0.8 / max(len(dens), 1)
    for k, d in enumerate(dens):
        mu, se = [], []
        for a in arms:
            v = df[(df.n_humans == d) & (df.config == a)][metric].values
            mu.append(np.mean(v) if len(v) else np.nan)
            se.append(np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        ax.bar(x + (k - (len(dens) - 1) / 2) * w, mu, width=w, yerr=se, capsize=3,
               color=PALETTE[k], edgecolor="white", linewidth=0.6,
               label=f"{d} pedestrians")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title or "Do the social terms bind only when the crowd binds?")
    ax.legend(fontsize=7.5)
    return ax


def plot_mechanism(df, ax=None, title=None):
    """Action autocorrelation against success, one point per run.

    The jerk hypothesis in one scatter: if R5's contribution runs through
    temporal action coherence, then every run that fails should sit at low
    autocorrelation regardless of which reward produced it, and any intervention
    that raises autocorrelation should move a run to the right AND up.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 3.8))
    for k, (lbl, g) in enumerate(df.groupby("group")):
        ax.scatter(g["action_autocorr"], g["success"], s=46, alpha=0.85,
                   color=PALETTE[k % len(PALETTE)], label=lbl,
                   edgecolor="white", linewidth=0.6)
    ax.set_xlabel("lag-1 action autocorrelation (evaluation)")
    ax.set_ylabel("success rate")
    ax.set_title(title or "Mechanism: temporal action coherence vs. task success")
    ax.legend(fontsize=7, loc="best")
    return ax


def plot_metric_profile(agg, arms, metrics, titles=None, ax=None,
                        reference=None):
    """z-scored deltas vs the reference, arms x metrics, as a heatmap.

    Lets a reader see in one glance that (say) R3 and R4 move the SOCIAL metrics
    without moving success, which a success-rate-only table hides completely.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(1.05 * len(metrics) + 3.2,
                                      0.34 * len(arms) + 1.8))
    idx = agg.set_index(agg["config"].astype(str))
    M = np.full((len(arms), len(metrics)), np.nan)
    for i, a in enumerate(arms):
        for j, m in enumerate(metrics):
            col = f"{m}_mean"
            if a in idx.index and col in idx.columns:
                M[i, j] = idx.loc[a, col]
    if reference is not None and reference in idx.index:
        ref = np.array([idx.loc[reference, f"{m}_mean"] if f"{m}_mean" in idx.columns
                        else np.nan for m in metrics])
        with np.errstate(invalid="ignore"):
            sd = np.nanstd(M, axis=0)
            sd = np.where(sd > 1e-12, sd, 1.0)
            M = (M - ref[None, :]) / sd[None, :]
    vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([(titles or {}).get(m, m.replace("_", " ")) for m in metrics],
                       rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms, fontsize=7.5)
    ax.grid(False)
    for i in range(len(arms)):
        for j in range(len(metrics)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.1f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(M[i, j]) > 0.6 * vmax else "0.15")
    ax.set_title("Behavioural profile (s.d. units vs. the full-reward reference)",
                 fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    return ax
