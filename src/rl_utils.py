"""
Training / evaluation / statistics utilities for the crowd-navigation reward
ablation.  Kept in a module (rather than notebook globals) so that every object
is importable by SubprocVecEnv worker processes under any multiprocessing
start method.
"""
import os, json, time, random, math
from collections import deque

import numpy as np
import pandas as pd
import gymnasium as gym

import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from crowd_env import (CrowdNavAblationEnv, make_cfg, ABLATION_CONFIGS,
                       orca_robot_action, resolve_arm, display_name)

METRIC_KEYS = [
    "success", "collision", "timeout", "nav_time", "path_length_ratio",
    "intrusion_steps", "intrusion_count", "intrusion_rate",
    "mean_sq_accel", "min_separation",
    # behavioural instrumentation -- these are what the mechanism analysis in
    # section 13 runs on.  action_autocorr is the direct measurement of the
    # temporal action coherence that R5 is hypothesised to supply;
    # displacement_efficiency separates "went nowhere" from "took a long route".
    "action_autocorr", "mean_speed", "displacement_efficiency",
    "net_progress_per_step",
]


# --------------------------------------------------------------------------- #
#  Reproducibility
# --------------------------------------------------------------------------- #
def set_global_seeds(seed: int, deterministic_torch: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    if deterministic_torch:
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- #
#  Callback: harvest per-episode metrics from the vectorised envs
# --------------------------------------------------------------------------- #
class EpisodeMetricsCallback(BaseCallback):
    def __init__(self, csv_path: str, rollout_csv_path: str = None,
                 window: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.rollout_csv_path = rollout_csv_path
        self.rows = []
        self.rollout_rows = []
        self._buf = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            m = info.get("episode_metrics")
            if m is not None:
                self.rows.append({"timesteps": int(self.num_timesteps), **m})
                self._buf.append(m)
        return True

    def _on_rollout_end(self) -> None:
        # Policy standard deviation is the single most diagnostic scalar for the
        # exploration-collapse / exploration-runaway failure modes.  Logging it
        # per rollout makes those failures visible instead of inferable.
        try:
            std = float(th.exp(self.model.policy.log_std.detach()).mean().item())
        except Exception:
            std = float("nan")
        self.logger.record("crowdnav/policy_std", std)

        row = {"timesteps": int(self.num_timesteps), "policy_std": std,
               "n_episodes_in_window": len(self._buf)}
        for k in METRIC_KEYS:
            vals = [b[k] for b in self._buf if np.isfinite(b[k])]
            if vals:
                m = float(np.mean(vals))
                self.logger.record(f"crowdnav/{k}", m)
                row[k] = m
        self.rollout_rows.append(row)

    def _on_training_end(self) -> None:
        self.flush()

    def flush(self):
        if self.rows:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            pd.DataFrame(self.rows).to_csv(self.csv_path, index=False)
        if self.rollout_rows and self.rollout_csv_path:
            pd.DataFrame(self.rollout_rows).to_csv(self.rollout_csv_path, index=False)


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #
def pick_device(requested="auto"):
    """MlpPolicy inference on a 40-D observation is launch-latency bound: the
    CPU->GPU->CPU round trip per rollout step costs more than the matmul saves.
    SB3 warns about exactly this.  'auto' therefore means CPU here."""
    if requested != "auto":
        return requested
    return "cpu"


def linear_schedule(initial: float, final_frac: float = 0.05):
    """PPO learning-rate schedule: `progress_remaining` goes 1 -> 0.

    Decaying the step size is the single cheapest way to buy a sharper final
    deterministic policy out of a fixed step budget, and it is applied
    identically to every arm so it cannot bias the ablation.  It matters more
    here than usual because the headline metric is evaluated with
    `deterministic=True`, which reads the MEAN of the action distribution: a
    policy still taking 3e-4 steps at the end of training has a mean that is
    still wandering.
    """
    class _LinearSchedule:
        """A callable with a readable repr, so the run's meta.json records the
        actual schedule rather than a memory address."""

        def __init__(self, initial, final_frac):
            self.initial, self.final_frac = float(initial), float(final_frac)

        def __call__(self, progress_remaining: float) -> float:
            return self.initial * (self.final_frac +
                                   (1.0 - self.final_frac) * progress_remaining)

        def __repr__(self):
            return (f"linear_schedule(initial={self.initial:g}, "
                    f"final={self.initial*self.final_frac:g})")

    return _LinearSchedule(initial, final_frac)


def default_ppo_kwargs(n_envs: int, gamma: float = 0.99, lr: float = 3e-4,
                       decay_lr: bool = True, net_width: int = 128):
    """Identical for every ablation configuration -- only the reward changes."""
    return dict(
        policy="MlpPolicy",
        learning_rate=linear_schedule(lr) if decay_lr else lr,
        n_steps=max(128, 4096 // n_envs),      # 4096-step rollout regardless of n_envs
        batch_size=512,
        n_epochs=10,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        # ent_coef MUST be 0 here.  A positive entropy bonus pushes log_std up
        # with a constant gradient (+2*ent_coef for a 2-D Gaussian).  The jerk
        # penalty is the only reward term that opposes it everywhere in the state
        # space -- E[-w_jerk*||a_t - a_{t-1}||^2] = -4*w_jerk*sigma^2 for i.i.d.
        # actions -- so with ent_coef>0 the jerk arms equilibrate at
        # sigma = sqrt(ent_coef / (4*w_jerk)) while the no-jerk arms have NOTHING
        # opposing the bonus and their sigma runs away.  That turns "ablate R5"
        # into "ablate entropy regularisation", which is a different experiment.
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        # --- exploration -----------------------------------------------------
        # v3 enabled gSDE to give the no-jerk arms temporal coherence from the
        # ALGORITHM rather than from R5.  It did not work: A5 and B1 still scored
        # exactly 0.000, while mean_sq_accel went 2.15 -> 28.3 (A5), 0.69 -> 8.11
        # (B1) and 1.15 -> 27.5 (B2), and B2 collapsed from 0.753 to 0.000.  A
        # failed intervention that costs 40% of navigation speed everywhere is
        # not worth carrying, so v4 reverts to i.i.d. Gaussian exploration and
        # instead breaks the exploration/smoothness confound experimentally, with
        # the B3 arm and the gSDE x jerk grid (see the notebook).
        use_sde=False,
        # Guards against the late-training policy collapse that PPO exhibits once
        # a run starts reliably terminating episodes and the advantage scale
        # shifts.  Applied identically to every arm.
        target_kl=0.03,
        policy_kwargs=dict(
            net_arch=dict(pi=[net_width, net_width],
                          vf=[net_width, net_width]),
            activation_fn=nn.ReLU,
            log_std_init=-0.5,
            ortho_init=True,
        ),
    )


def train_one(config_name, seed, total_timesteps, n_envs, out_root,
              device="auto", env_overrides=None, verbose=0, net_width=128,
              run_tag=None):
    """Train a single (reward-configuration, seed) run. Returns a summary dict."""
    config_name = resolve_arm(config_name)
    run_id = f"{run_tag or config_name}__seed{seed}"
    run_dir = os.path.join(out_root, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    set_global_seeds(seed)
    # One torch thread per process.  With SubprocVecEnv workers already
    # saturating the vCPUs, letting torch spawn its own OMP pool for a tiny MLP
    # causes heavy context-switching: it was the main reason measured throughput
    # (1.1k steps/s) came in 5x under the env-only projection.
    th.set_num_threads(1)
    # Torch's own pool is not the only one: numpy/BLAS spawn theirs at import
    # time, which set_num_threads cannot reach.  The notebook sets the OMP_*
    # environment variables before the first import for that reason.
    cfg = make_cfg(config_name, **(env_overrides or {}))

    # DummyVecEnv, always.  SubprocVecEnv pickles a 40-float observation across
    # a pipe and synchronises on the slowest worker EVERY step; measured in v3 at
    # 1,254 steps/s against 3,850 steps/s for a single un-vectorised env, i.e.
    # four workers delivering one third of one env.  Stepping four envs in-process
    # costs 4x the env work and zero IPC, and parallelism is recovered at the RUN
    # level instead (train_many), where the sync barrier is one join per run.
    vec_cls = DummyVecEnv
    env = make_vec_env(
        CrowdNavAblationEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=dict(cfg=cfg),
        monitor_dir=os.path.join(run_dir, "monitor"),
        vec_env_cls=vec_cls,
    )

    ppo_kwargs = default_ppo_kwargs(n_envs, gamma=cfg.gamma, net_width=net_width)
    # The PBRS invariance guarantee (Ng et al., 1999) only holds when the
    # discount used to build the shaping term equals the agent's discount.
    assert abs(ppo_kwargs["gamma"] - cfg.gamma) < 1e-12

    # TensorBoard logging is a convenience, never a dependency.  SB3 raises at
    # construction time if `tensorboard_log` is set and the package is missing,
    # which would otherwise turn "pip could not reach the internet" into "the
    # entire sweep failed".  Per-run metrics are written to CSV regardless.
    try:
        import tensorboard  # noqa: F401
        _tb_log = os.path.join(out_root, "tb")
    except Exception:
        _tb_log = None
    model = PPO(env=env, seed=seed, device=pick_device(device), verbose=verbose,
                tensorboard_log=_tb_log, **ppo_kwargs)

    cb = EpisodeMetricsCallback(os.path.join(run_dir, "train_episodes.csv"),
                                os.path.join(run_dir, "train_rollouts.csv"))
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=cb,
                tb_log_name=run_id, progress_bar=False)
    wall = time.time() - t0

    model.save(os.path.join(run_dir, "model"))
    env.close()

    # ppo_kwargs holds objects json cannot represent -- `learning_rate` is a
    # schedule CALLABLE and `policy_kwargs` contains a torch activation class.
    # Serialise defensively: anything not JSON-native becomes its repr.  Getting
    # this wrong is expensive rather than obvious, because model.save() has
    # already succeeded by this point, so the run leaves a usable model.zip next
    # to a TRUNCATED meta.json -- and the resume path then reads that truncated
    # file on the next attempt.
    def _jsonable(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (list, tuple)):
            return [_jsonable(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _jsonable(x) for k, x in v.items()}
        return repr(v)

    meta = dict(run_id=run_id, config=config_name, seed=int(seed),
                net_width=int(net_width),
                total_timesteps=int(total_timesteps), n_envs=int(n_envs),
                wall_seconds=round(wall, 1),
                fps=round(total_timesteps / max(wall, 1e-6), 1),
                ppo_kwargs={k: _jsonable(v) for k, v in ppo_kwargs.items()})
    # Write atomically so an interrupted session can never leave a half-written
    # meta.json for the resume path to choke on.
    _mp = os.path.join(run_dir, "meta.json")
    with open(_mp + ".tmp", "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(_mp + ".tmp", _mp)
    return meta


# --------------------------------------------------------------------------- #
#  Deterministic evaluation on a FIXED, shared set of test scenarios
# --------------------------------------------------------------------------- #
class SeededEvalEnv(gym.Wrapper):
    """Replays a fixed list of scenario seeds, one per reset, cycling forever."""

    def __init__(self, env, seeds):
        super().__init__(env)
        self.seeds = [int(s) for s in seeds]
        self.i = 0

    def reset(self, **kwargs):
        s = self.seeds[self.i % len(self.seeds)]
        self.i += 1
        return self.env.reset(seed=s)


def evaluate_fixed(model, config_name, test_seeds, n_workers=8,
                   env_overrides=None, deterministic=True, max_iters=100000):
    """Evaluate a policy on exactly `test_seeds` scenarios.

    Every configuration and seed sees the *same* scenarios, which removes
    scenario sampling noise from the between-config comparison.
    """
    test_seeds = list(test_seeds)
    n_workers = max(1, min(n_workers, len(test_seeds)))
    chunks = [test_seeds[i::n_workers] for i in range(n_workers)]

    def _mk(ch):
        def _f():
            return SeededEvalEnv(
                CrowdNavAblationEnv(make_cfg(config_name, **(env_overrides or {}))), ch)
        return _f

    vec = DummyVecEnv([_mk(ch) for ch in chunks])
    obs = vec.reset()

    targets = [len(c) for c in chunks]
    done_counts = [0] * n_workers
    running_return = np.zeros(n_workers)
    records = []

    it = 0
    while any(done_counts[i] < targets[i] for i in range(n_workers)):
        actions, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = vec.step(actions)
        running_return += np.asarray(rewards, dtype=np.float64)
        for i, info in enumerate(infos):
            m = info.get("episode_metrics")
            if m is None:
                continue
            if done_counts[i] < targets[i]:
                rec = dict(m)
                rec["scenario_seed"] = chunks[i][done_counts[i]]
                rec["ep_return"] = float(running_return[i])
                records.append(rec)
                done_counts[i] += 1
            running_return[i] = 0.0
        it += 1
        if it > max_iters:
            break
    vec.close()
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
#  Statistics
# --------------------------------------------------------------------------- #
def summarise_run(df: pd.DataFrame) -> dict:
    """Collapse per-episode evaluation records into one row for a run."""
    out = {}
    for k in METRIC_KEYS:
        if k not in df.columns:
            continue
        v = df[k].replace([np.inf, -np.inf], np.nan).dropna()
        out[k] = float(v.mean()) if len(v) else np.nan
    # Conditioning some metrics on successful episodes only is standard in the
    # crowd-navigation literature (nav-time / path-ratio are undefined otherwise).
    succ = df[df["success"] > 0.5]
    out["nav_time_succ"] = float(succ["nav_time"].mean()) if len(succ) else np.nan
    out["path_length_ratio_succ"] = float(succ["path_length_ratio"].mean()) if len(succ) else np.nan
    out["mean_sq_accel_succ"] = float(succ["mean_sq_accel"].mean()) if len(succ) else np.nan
    out["ep_return"] = float(df["ep_return"].mean()) if "ep_return" in df else np.nan
    out["n_episodes"] = int(len(df))
    return out


def aggregate_across_seeds(per_run: pd.DataFrame, metrics, alpha=0.05):
    """mean, sd and Student-t (1-alpha) CI half-width across training seeds."""
    from scipy import stats
    rows = []
    for cfg, g in per_run.groupby("config"):
        row = {"config": cfg, "n_seeds": len(g)}
        for m in metrics:
            v = g[m].replace([np.inf, -np.inf], np.nan).dropna().values
            n = len(v)
            mu = float(np.mean(v)) if n else np.nan
            sd = float(np.std(v, ddof=1)) if n > 1 else 0.0
            hw = (stats.t.ppf(1 - alpha / 2, n - 1) * sd / math.sqrt(n)) if n > 1 else np.nan
            row[f"{m}_mean"] = mu
            row[f"{m}_sd"] = sd
            row[f"{m}_ci95"] = hw
        rows.append(row)
    return pd.DataFrame(rows).sort_values("config").reset_index(drop=True)


def paired_tests(per_episode: pd.DataFrame, reference: str, metrics):
    """Compare each ablation against the reference config on the shared test set.

    Episodes are paired by scenario seed and pooled across training seeds, so a
    Wilcoxon signed-rank test on the per-scenario means is appropriate.
    """
    from scipy import stats
    ref = (per_episode[per_episode.config == reference]
           .groupby("scenario_seed")[metrics].mean())
    rows = []
    for cfg, g in per_episode.groupby("config"):
        if cfg == reference:
            continue
        cur = g.groupby("scenario_seed")[metrics].mean()
        idx = ref.index.intersection(cur.index)
        row = {"config": cfg, "reference": reference, "n_paired": len(idx)}
        for m in metrics:
            a = ref.loc[idx, m].values
            b = cur.loc[idx, m].values
            d = b - a
            row[f"{m}_delta"] = float(np.mean(d))
            if np.allclose(d, 0):
                row[f"{m}_p"] = 1.0
            else:
                try:
                    row[f"{m}_p"] = float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)
                except Exception:
                    row[f"{m}_p"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def holm_bonferroni(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


# --------------------------------------------------------------------------- #
#  ONNX export (PPO / ActorCriticPolicy -- note: PPO has NO `.actor` attribute)
# --------------------------------------------------------------------------- #
class OnnxDeterministicPolicy(nn.Module):
    """Deterministic actor forward pass, stripped of value head and sampling."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> th.Tensor:
        features = self.policy.extract_features(observation)
        if isinstance(features, tuple):
            features = features[0]
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        mean_actions = self.policy.action_net(latent_pi)
        return th.clamp(mean_actions, -1.0, 1.0)


def export_onnx(model, path, obs_dim, opset=17):
    """Export the deterministic actor. Tries the TorchScript exporter first and
    falls back to the dynamo exporter on newer PyTorch builds."""
    model.policy.set_training_mode(False)
    wrapper = OnnxDeterministicPolicy(model.policy).to("cpu").eval()
    dummy = th.zeros(1, obs_dim, dtype=th.float32)
    kw = dict(opset_version=opset, input_names=["obs"], output_names=["action"],
              dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}})
    try:
        th.onnx.export(wrapper, dummy, path, dynamo=False, **kw)
    except TypeError:
        th.onnx.export(wrapper, dummy, path, **kw)          # older PyTorch
    except Exception:
        th.onnx.export(wrapper, dummy, path, dynamo=True, **kw)
    return path


def verify_onnx(model, onnx_path, obs_dim, n=512, tol=1e-4, seed=0):
    """Numerical parity check: ONNX graph vs. SB3 `predict(deterministic=True)`."""
    import onnxruntime as ort
    rng = np.random.default_rng(seed)
    obs = rng.normal(0, 0.6, size=(n, obs_dim)).astype(np.float32)
    sb3_act, _ = model.predict(obs, deterministic=True)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_act = sess.run(["action"], {"obs": obs})[0]
    err = float(np.max(np.abs(sb3_act - onnx_act)))
    return err, bool(err < tol)


def crossvalidate_orca(n_trials=200, n_agents=6, tol=2e-4, seed=0):
    """Compare the bundled pure-Python ORCA against the reference `python-rvo2`
    Cython bindings on random scenes.  Returns (max_abs_error, n_compared) or
    None when python-rvo2 is not installed."""
    try:
        import rvo2
    except Exception:
        return None
    from orca import orca_velocity
    rng = np.random.default_rng(seed)
    dt, horizon, ndist, maxn = 0.25, 5.0, 10.0, 10
    worst, count = 0.0, 0
    for _ in range(n_trials):
        pos = rng.uniform(-5, 5, size=(n_agents, 2))
        vel = rng.uniform(-1, 1, size=(n_agents, 2))
        rad = rng.uniform(0.3, 0.5, size=n_agents)
        pref = rng.uniform(-1, 1, size=(n_agents, 2))
        sim = rvo2.PyRVOSimulator(dt, ndist, maxn, horizon, horizon, 0.3, 1.0)
        ids = []
        for i in range(n_agents):
            a = sim.addAgent(tuple(pos[i]), ndist, maxn, horizon, horizon,
                             float(rad[i]), 1.0, tuple(vel[i]))
            sim.setAgentPrefVelocity(a, tuple(pref[i]))
            ids.append(a)
        sim.doStep()
        for i in range(n_agents):
            ref = np.asarray(sim.getAgentVelocity(ids[i]))
            neigh = [(tuple(pos[j]), tuple(vel[j]), float(rad[j]))
                     for j in range(n_agents) if j != i]
            mine = np.asarray(orca_velocity(
                tuple(pos[i]), tuple(vel[i]), float(rad[i]), tuple(pref[i]), 1.0,
                neigh, time_horizon=horizon, time_step=dt,
                neighbour_dist=ndist, max_neighbours=maxn, responsibility=0.5))
            worst = max(worst, float(np.max(np.abs(ref - mine))))
            count += 1
    return worst, count


# --------------------------------------------------------------------------- #
#  Training-health diagnostics
# --------------------------------------------------------------------------- #
def training_health(runs_root, configs, seeds, metric="success", tail_frac=0.15):
    """Detect late-training collapse and exploration runaway from the rollout logs.

    `peak` is the best windowed value reached at any point; `final` is the mean
    over the last `tail_frac` of training.  A large peak-to-final drop is the
    signature of PPO collapse -- it looks nothing like "never learned", which
    shows up as a flat low curve with peak ~ final.
    """
    rows = []
    for cfg in configs:
        for sd in seeds:
            f = os.path.join(runs_root, f"{cfg}__seed{sd}", "train_rollouts.csv")
            if not os.path.exists(f):
                continue
            df = pd.read_csv(f)
            if metric not in df or df.empty:
                continue
            v = df[metric].rolling(5, min_periods=1).mean().values
            k = max(1, int(len(v) * tail_frac))
            peak, final = float(np.nanmax(v)), float(np.nanmean(v[-k:]))
            std0 = float(df["policy_std"].iloc[0]) if "policy_std" in df else np.nan
            std1 = float(df["policy_std"].iloc[-1]) if "policy_std" in df else np.nan
            rows.append({
                "config": cfg, "seed": sd,
                f"{metric}_peak": peak, f"{metric}_final": final,
                "drop": peak - final,
                "collapsed": bool(peak > 0.25 and (peak - final) > 0.3 * peak),
                "policy_std_start": std0, "policy_std_end": std1,
                "std_ratio": (std1 / std0) if std0 and np.isfinite(std0) else np.nan,
                "runaway_exploration": bool(np.isfinite(std1) and np.isfinite(std0)
                                            and std1 > 1.5 * std0),
            })
    cols = ["config", "seed", f"{metric}_peak", f"{metric}_final", "drop", "collapsed",
            "policy_std_start", "policy_std_end", "std_ratio", "runaway_exploration"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


# --------------------------------------------------------------------------- #
#  Run-level parallelism
# --------------------------------------------------------------------------- #
def _train_worker(kw):
    """Module-level so it is picklable by the spawn start method."""
    import os as _os
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        _os.environ[v] = "1"
    return train_one(**kw)


def train_many(plan, total_timesteps, n_envs, out_root, device="auto",
               env_overrides=None, n_parallel=None, verbose=True,
               net_width=128, run_tag=None):
    """Train `plan` = [(config_name, seed), ...] with run-level parallelism.

    Each worker owns one PPO run and one in-process DummyVecEnv, so the only
    synchronisation in the whole sweep is a join per finished run.  Runs whose
    model.zip already exists are skipped, which is what makes the sweep
    resumable across Kaggle sessions.
    """
    import multiprocessing as mp, time as _time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_parallel = n_parallel or max(1, (os.cpu_count() or 4))
    todo, meta = [], []
    for cfg_name, seed in plan:
        tag = run_tag or resolve_arm(cfg_name)
        run_dir = os.path.join(out_root, "runs", f"{tag}__seed{seed}")
        mpath = os.path.join(run_dir, "meta.json")
        if os.path.exists(os.path.join(run_dir, "model.zip")) and os.path.exists(mpath):
            try:
                meta.append(json.load(open(mpath)))
                if verbose:
                    print(f"SKIP (already trained) {cfg_name} seed={seed}")
                continue
            except Exception:
                # A corrupt meta.json means the previous attempt died between
                # saving the model and writing its metadata.  The model itself is
                # fine, so record what we know and keep the run rather than
                # retraining it or killing the sweep.
                meta.append({"run_id": f"{tag}__seed{seed}", "config": resolve_arm(cfg_name),
                             "seed": int(seed), "total_timesteps": int(total_timesteps),
                             "n_envs": int(n_envs), "net_width": int(net_width),
                             "wall_seconds": float("nan"), "fps": float("nan"),
                             "recovered_from_corrupt_meta": True})
                if verbose:
                    print(f"SKIP (already trained, meta recovered) {cfg_name} seed={seed}")
                continue
        todo.append(dict(config_name=cfg_name, seed=seed,
                         total_timesteps=total_timesteps, n_envs=n_envs,
                         out_root=out_root, device=device,
                         env_overrides=env_overrides, verbose=0,
                         net_width=net_width, run_tag=tag))
    if not todo:
        return meta

    t0 = _time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_parallel, mp_context=ctx) as ex:
        futs = {ex.submit(_train_worker, kw): kw for kw in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            kw = futs[fut]
            try:
                m = fut.result()
                meta.append(m)
                if verbose:
                    el = _time.time() - t0
                    eta = el / i * (len(todo) - i) / 60
                    print(f"[{i:2d}/{len(todo)}] {m['config']:20s} seed={m['seed']}  "
                          f"{m['wall_seconds']/60:5.1f} min  {m['fps']:6.0f} fps   "
                          f"ETA {eta:5.1f} min")
            except Exception as e:                       # one bad run must not
                print(f"!! run failed: {kw['config_name']} "  # kill the sweep
                      f"seed={kw['seed']}: {type(e).__name__}: {e}")
    if verbose:
        print(f"\ntrained {len(todo)} runs in {(_time.time()-t0)/60:.1f} min "
              f"on {n_parallel} workers")
    return meta


# --------------------------------------------------------------------------- #
#  Reward-landscape audit  (no training required -- run this BEFORE training)
# --------------------------------------------------------------------------- #
def reward_landscape_audit(configs, seeds, env_overrides=None,
                           safety_space=0.45, n_episodes=120):
    """Compare the episode return of a COMPETENT reference policy against
    STANDING STILL, under each arm's own reward.

    A negative margin means the arm's global optimum is to freeze -- no amount
    of training or tuning will produce a navigating policy, because navigating
    is genuinely worse under that reward.  Finding this out in 30 seconds beats
    finding it out after a six-hour sweep.
    """
    competent = lambda e: orca_robot_action(e, 1.0, safety_space)
    freeze = lambda e: np.zeros(2)
    ep_seeds = list(seeds)[:n_episodes]
    rows = []
    for arm in configs:
        cfg = make_cfg(arm, **(env_overrides or {}))
        out = {}
        for label, pol in (("competent", competent), ("freeze", freeze)):
            rets, succ = [], []
            env = CrowdNavAblationEnv(cfg)
            for s in ep_seeds:
                env.reset(seed=int(s)); R = 0.0
                while True:
                    _, r, te, tu, info = env.step(pol(env))
                    R += r
                    if te or tu: break
                rets.append(R); succ.append(info["episode_metrics"]["success"])
            out[label] = float(np.mean(rets))
            out[label + "_success"] = float(np.mean(succ))
        rows.append({"config": arm, **out,
                     "margin": out["competent"] - out["freeze"],
                     "freeze_is_optimal": bool(out["competent"] <= out["freeze"])})
    return pd.DataFrame(rows)


def exploration_reachability(env_overrides=None, n_episodes=200, sigma=0.6):
    """How often does an UNTRAINED policy reach the goal region, as a function
    of the temporal correlation of its exploration noise?  This is what decides
    whether sparse-reward arms can bootstrap at all."""
    cfg = make_cfg("R1-5_full", **(env_overrides or {}))
    rows = []
    for label, tau in (("i.i.d. (tau=1)", 1), ("correlated tau=8", 8),
                       ("correlated tau=16", 16), ("correlated tau=50", 50)):
        rng = np.random.default_rng(1)
        env = CrowdNavAblationEnv(cfg)
        hit, disp, near = 0.0, [], []
        rho = math.exp(-1.0 / tau)
        for s in range(9000, 9000 + n_episodes):
            env.reset(seed=s); p0 = env.robot_pos.copy()
            a = np.zeros(2); md = np.inf
            while True:
                a = rho * a + math.sqrt(1 - rho ** 2) * rng.normal(0, sigma, 2)
                _, _, te, tu, info = env.step(np.clip(a, -1, 1))
                md = min(md, float(np.linalg.norm(env.robot_pos - env.robot_goal)))
                if te or tu: break
            hit += info["episode_metrics"]["success"]
            disp.append(float(np.linalg.norm(env.robot_pos - p0))); near.append(md)
        rows.append({"exploration": label, "goal_reach_rate": hit / n_episodes,
                     "net_displacement_m": float(np.mean(disp)),
                     "closest_approach_m": float(np.mean(near)),
                     "frac_within_2m": float(np.mean(np.array(near) < 2.0))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Pre-flight: short training run on the arms most likely to fail
# --------------------------------------------------------------------------- #
def preflight(configs, out_root, timesteps=150_000, n_envs=4, seed=0,
              env_overrides=None, device="auto", net_width=128):
    """Train the riskiest arms briefly and report whether they get off zero.

    Returns (DataFrame, measured_fps).  The measured fps is the ONLY honest
    input to the time budget: extrapolating single-env throughput by the worker
    count ignores IPC and scheduler contention and overestimated by 5x.
    """
    rows, total_steps, t0 = [], 0, time.time()
    for arm in configs:
        run_dir = os.path.join(out_root, "preflight", arm)
        os.makedirs(run_dir, exist_ok=True)
        set_global_seeds(seed); th.set_num_threads(1)
        cfg = make_cfg(arm, **(env_overrides or {}))
        vec_cls = DummyVecEnv          # see train_one
        env = make_vec_env(CrowdNavAblationEnv, n_envs=n_envs, seed=seed,
                           env_kwargs=dict(cfg=cfg), vec_env_cls=vec_cls)
        model = PPO(env=env, seed=seed, device=pick_device(device), verbose=0,
                    **default_ppo_kwargs(n_envs, gamma=cfg.gamma, net_width=net_width))
        cb = EpisodeMetricsCallback(os.path.join(run_dir, "train_episodes.csv"),
                                    os.path.join(run_dir, "train_rollouts.csv"))
        model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
        cb.flush(); env.close()
        total_steps += timesteps

        df = pd.DataFrame(cb.rollout_rows)
        peak = float(df["success"].max()) if "success" in df and len(df) else 0.0
        last = float(df["success"].iloc[-3:].mean()) if "success" in df and len(df) else 0.0
        rows.append({"config": arm, "steps": timesteps, "peak_success": peak,
                     "final_success": last,
                     "policy_std_end": float(df["policy_std"].iloc[-1]) if len(df) else np.nan,
                     "learning": bool(peak > 0.02)})
        del model
    fps = total_steps / max(time.time() - t0, 1e-6)
    return pd.DataFrame(rows), fps


# --------------------------------------------------------------------------- #
#  Bimodal-safe statistics
# --------------------------------------------------------------------------- #
def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0: return np.nan, np.nan, np.nan
    if len(v) == 1: return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def aggregate_robust(per_run: pd.DataFrame, metrics, solved_threshold=0.5, alpha=0.05):
    """Across-seed aggregation that survives bimodal outcomes.

    Student-t CIs assume unimodality.  When an arm gives 0.94/0.92/0.00/0.91/0.98
    the t-interval is both enormous and meaningless.  We report the bootstrap
    percentile interval, the median with IQR, and -- most informative of all --
    the fraction of seeds that solved the task at all.
    """
    rows = []
    if per_run is None or len(per_run) == 0 or "config" not in per_run.columns:
        cols = ["config", "n_seeds", "seeds_solved"]
        for m in metrics:
            cols += [f"{m}_{k}" for k in ("mean", "lo", "hi", "ci95", "median", "iqr")]
        return pd.DataFrame(columns=cols)
    for cfg, g in per_run.groupby("config"):
        row = {"config": cfg, "n_seeds": len(g)}
        if "success" in g:
            row["seeds_solved"] = float((g["success"] > solved_threshold).mean())
        for m in metrics:
            v = pd.to_numeric(g[m], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
            mu, lo, hi = bootstrap_ci(v, alpha=alpha)
            row[f"{m}_mean"] = mu
            row[f"{m}_lo"] = lo
            row[f"{m}_hi"] = hi
            row[f"{m}_ci95"] = (hi - lo) / 2 if np.isfinite(hi) else np.nan
            row[f"{m}_median"] = float(np.median(v)) if len(v) else np.nan
            row[f"{m}_iqr"] = float(np.subtract(*np.percentile(v, [75, 25]))) if len(v) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------- #
#  Deadline-aware budget scheduler
# --------------------------------------------------------------------------- #
#  A Kaggle session dies at a hard wall-clock limit.  A sweep that is 80% done
#  when that happens has produced nothing analysable, so the budget cannot be a
#  comment in the config cell -- it has to be an object that every stage asks
#  before it starts and that sizes the stage from the time that is ACTUALLY left.
class Budget:
    """Tracks wall clock against a hard deadline and sizes each stage from it."""

    def __init__(self, total_hours, reserve_hours=0.0, label="session"):
        self.t0 = time.time()
        self.total = float(total_hours) * 3600.0
        self.reserve = float(reserve_hours) * 3600.0
        self.label = label
        self.log = []

    # -- clock ---------------------------------------------------------------
    @property
    def elapsed(self):
        return time.time() - self.t0

    @property
    def remaining(self):
        """Seconds left before the deadline, minus the post-training reserve."""
        return self.total - self.reserve - self.elapsed

    def hours_left(self):
        return self.remaining / 3600.0

    def set_reserve(self, hours):
        self.reserve = float(hours) * 3600.0

    # -- sizing --------------------------------------------------------------
    def steps_for(self, n_runs, fps, share=1.0, target=None,
                  lo=100_000, hi=1_000_000, quantum=10_000):
        """Largest per-run step count that fits `share` of the remaining budget.

        `fps` is the MEASURED aggregate throughput (steps/s across all workers),
        never an extrapolation from single-env speed -- v2 of this study
        overestimated by 5x doing exactly that and had to be re-run.
        """
        if n_runs <= 0 or fps <= 0:
            return 0
        affordable = int(self.remaining * share * fps / n_runs)
        s = affordable if target is None else min(target, affordable)
        s = int(s // quantum * quantum)
        return int(max(lo, min(hi, s))) if s >= lo else 0

    def can_afford(self, n_runs, steps, fps, safety=1.15):
        """Would this stage fit, with a safety factor on the estimate?"""
        if fps <= 0:
            return False
        return (n_runs * steps / fps) * safety <= self.remaining

    def eta_hours(self, n_runs, steps, fps):
        return (n_runs * steps / fps) / 3600.0 if fps > 0 else float("inf")

    # -- reporting -----------------------------------------------------------
    def stage(self, name, n_runs, steps, fps):
        eta = self.eta_hours(n_runs, steps, fps)
        ok = self.can_afford(n_runs, steps, fps)
        print(f"  [budget] {name:26s} {n_runs:4d} runs x {steps:>8,} steps "
              f"-> {eta:5.2f} h   left {self.hours_left():5.2f} h   "
              f"{'GO' if ok else 'SKIP (would overrun)'}")
        return ok

    def mark(self, name):
        self.log.append({"stage": name, "elapsed_h": round(self.elapsed / 3600, 3),
                         "remaining_h": round(self.hours_left(), 3)})
        print(f"  [budget] {name:26s} done at {self.elapsed/3600:5.2f} h "
              f"| {self.hours_left():5.2f} h left")

    def frame(self):
        return pd.DataFrame(self.log)


# --------------------------------------------------------------------------- #
#  Evaluating a whole stage
# --------------------------------------------------------------------------- #
def evaluate_stage(arms, seeds, out_root, test_seeds, env_overrides,
                   n_workers=10, verbose=True, label=""):
    """Load every trained model in a stage and evaluate it on the shared test set.

    Returns (per_run DataFrame, per_episode DataFrame).  Missing models are
    skipped rather than raising, so a stage that was cut short by the budget
    guard still produces an analysable table for the runs that did finish.
    """
    from stable_baselines3 import PPO
    rows, eps = [], []
    t0 = time.time()
    for arm in arms:
        for sd in seeds:
            mp = os.path.join(out_root, "runs", f"{resolve_arm(arm)}__seed{sd}",
                              "model.zip")
            if not os.path.exists(mp):
                continue
            model = PPO.load(mp, device="cpu")
            df = evaluate_fixed(model, arm, test_seeds, n_workers=n_workers,
                                env_overrides=env_overrides, deterministic=True)
            df["config"] = resolve_arm(arm)
            df["train_seed"] = sd
            eps.append(df)
            s = summarise_run(df)
            rows.append({"config": resolve_arm(arm), "seed": sd, **s})
            del model
    per_run = (pd.DataFrame(rows) if rows else
               pd.DataFrame(columns=["config", "seed"] + METRIC_KEYS))
    per_ep = pd.concat(eps, ignore_index=True) if eps else pd.DataFrame()
    if verbose:
        print(f"  evaluated {len(per_run)} runs x {len(test_seeds)} scenarios "
              f"({len(per_ep):,} episodes) in {(time.time()-t0)/60:.1f} min  {label}")
    return per_run, per_ep


def solved_fraction(per_run, arm, threshold=0.5):
    g = per_run[per_run.config == arm]
    return float((g["success"] > threshold).mean()) if len(g) else float("nan")


# --------------------------------------------------------------------------- #
#  Architecture pre-flight: equal WALL-CLOCK comparison, not equal step count
# --------------------------------------------------------------------------- #
class _WallClockStop(BaseCallback):
    """Stops learning after `seconds`, so two configurations can be compared on
    equal wall time rather than equal step count."""

    def __init__(self, seconds):
        super().__init__(0)
        self.seconds = float(seconds)
        self.t0 = None

    def _on_training_start(self):
        self.t0 = time.time()

    def _on_step(self):
        return (time.time() - self.t0) < self.seconds


def arch_preflight(arm, out_root, widths=(128, 256), seconds=180, n_envs=4,
                   seed=0, env_overrides=None, device="auto"):
    """Choose the policy width by MEASUREMENT instead of by taste.

    Both widths train the same arm for the same number of seconds.  The wider
    net does more work per environment step, so at equal wall clock it reaches
    fewer steps; whether that trade is worth it is an empirical question about
    this particular observation space, and it takes six minutes to answer.
    The winner is whichever reaches the higher peak windowed success -- i.e.
    more learning per second of the session's budget.
    """
    rows = []
    for w in widths:
        set_global_seeds(seed); th.set_num_threads(1)
        cfg = make_cfg(arm, **(env_overrides or {}))
        env = make_vec_env(CrowdNavAblationEnv, n_envs=n_envs, seed=seed,
                           env_kwargs=dict(cfg=cfg), vec_env_cls=DummyVecEnv)
        model = PPO(env=env, seed=seed, device=pick_device(device), verbose=0,
                    **default_ppo_kwargs(n_envs, gamma=cfg.gamma, net_width=w))
        run_dir = os.path.join(out_root, "preflight", f"arch{w}")
        os.makedirs(run_dir, exist_ok=True)
        cb = EpisodeMetricsCallback(os.path.join(run_dir, "train_episodes.csv"),
                                    os.path.join(run_dir, "train_rollouts.csv"))
        t0 = time.time()
        model.learn(total_timesteps=10 ** 9, progress_bar=False,
                    callback=[cb, _WallClockStop(seconds)])
        wall = time.time() - t0
        cb.flush(); env.close()
        df = pd.DataFrame(cb.rollout_rows)
        peak = float(df["success"].rolling(3, min_periods=1).mean().max()) \
            if "success" in df and len(df) else 0.0
        final = float(df["success"].iloc[-3:].mean()) if "success" in df and len(df) else 0.0
        rows.append({"net_width": w, "seconds": round(wall, 1),
                     "steps_reached": int(model.num_timesteps),
                     "fps": round(model.num_timesteps / max(wall, 1e-6), 1),
                     "peak_success": peak, "final_success": final})
        del model
    df = pd.DataFrame(rows)
    best = int(df.sort_values(["peak_success", "fps"], ascending=False).iloc[0]["net_width"])
    return df, best
