"""
CrowdNavAblationEnv -- a Gymnasium environment for crowd-aware navigation with a
fully modular, config-driven reward function.

Scenario follows the standard `circle crossing` benchmark used by CADRL /
CrowdNav / SARL (Chen et al., ICRA 2019): N humans are placed on a circle and
must cross to the antipodal point; the robot must traverse the same circle.
Humans are driven by ORCA and (by default) cannot see the robot -- the
"invisible robot" setting, which is the harder and more standard variant because
it forbids the policy from offloading collision avoidance onto the crowd.

The five ablatable reward components are implemented as independent, additively
combined terms so that a configuration is fully described by a dict of booleans.
"""
from dataclasses import dataclass, field, asdict
import math
import os
from dataclasses import replace as _dc_replace
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from orca import orca_velocity

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #

@dataclass
class EnvConfig:
    # --- scenario -----------------------------------------------------------
    n_humans: int = 5
    circle_radius: float = 4.0
    position_noise: float = 0.5      # jitter on the spawn circle
    robot_radius: float = 0.3
    human_radius_range: tuple = (0.3, 0.4)
    v_pref: float = 1.0              # robot preferred speed  (m/s)
    human_v_pref_range: tuple = (0.8, 1.2)
    kinematics: str = "holonomic"    # 'holonomic' | 'unicycle'
    action_frame: str = "goal_aligned"   # 'goal_aligned' | 'world' (holonomic only)
    omega_max: float = 1.5           # rad/s, unicycle only
    robot_visible: bool = False      # invisible-robot setting (harder, standard)

    # --- time ---------------------------------------------------------------
    time_step: float = 0.25          # s
    time_limit: float = 25.0         # s  -> 100 steps

    # --- ORCA (humans) ------------------------------------------------------
    orca_time_horizon: float = 5.0
    orca_neighbour_dist: float = 10.0
    human_goal_tolerance: float = 0.3

    # --- termination --------------------------------------------------------
    goal_tolerance: float = 0.3      # r_goal, distance at which success fires

    # --- reward weights (held CONSTANT across every ablation) ---------------
    w_goal: float = 1.0              # R1  terminal success bonus
    w_dist: float = 0.60             # R2  potential-based distance shaping
    w_space: float = 0.20            # R3  personal-space (proxemic) penalty
    w_ttc: float = 0.02              # R4  time-to-collision penalty
    w_jerk: float = 0.02             # R5  jerk / action-smoothness penalty
    # V4.0 shipped -0.75 and it was not enough.  PBRS sets Phi(terminal)=0 for
    # EVERY terminal state, so a terminating episode is refunded the whole
    # accumulated potential -- measured at +0.667 -- whether it ended at the goal
    # or inside a pedestrian.  A crash therefore cost 0.667-0.75 = -0.08 net.
    # Combined with r_step that made "dash and crash" (-0.442) tie with "stand
    # still" (-0.456), and crashing is the far easier basin to reach by gradient
    # descent, so the V4.0 quick run came back at collision = 1.00.
    # Measured over 80 held-out scenarios (competent / freeze / dash):
    #   -0.75, -0.010  ->  +0.406 / -0.464 / -0.474   dash ~ freeze  (the trap)
    #   -1.25, -0.005  ->  +0.467 / -0.226 / -0.880   clean ordering
    r_collision: float = -1.25       # task-level penalty, NOT ablated
    # Task-level per-step time cost.  NOT an ablatable component -- it is part
    # of the task definition, exactly like r_collision.  Its job is to remove
    # the freeze basin: with PBRS the shaping term F = gamma*Phi' - Phi expands
    # to gamma*progress/L + (1-gamma)*d/L, whose second half pays the agent
    # every step simply for standing far from the goal (+0.006/step in v3, i.e.
    # +0.6 over a 100-step timeout against a goal bonus of 1.0).  The v3 reward
    # landscape audit measured a FREEZE return of +0.406 under R1-5_full.
    # -0.01/step costs a full timeout -1.0 and drives that freeze return
    # strictly negative without touching the collision/shaping balance.
    # -0.005 rather than -0.010: with r_collision at -1.25 the freeze basin is
    # already gone (freeze = -0.226) and a smaller time cost leaves the competent
    # policy a healthy +0.467 instead of a thin +0.256.
    r_step: float = -0.005
    gamma: float = 0.99              # discount used inside the PBRS term

    dist_mode: str = "pbrs"          # 'pbrs' (policy-invariant) | 'progress' (naive)
    # Phi(s) = -||p_r - g|| / potential_scale.  The scale is NOT cosmetic.  With
    # an unnormalised potential the PBRS step reward contains a standing bonus
    #     F = [progress] + (1 - gamma) * |Phi(s')|
    # that pays the agent, every step, simply for being far from the goal.  At
    # scale=1, gamma=0.99, d=8 m and w_dist=0.15 that is +0.012/step -> +1.2 over
    # a 100-step timeout, exactly equal to the entire shaping budget earned by
    # reaching the goal.  Standing still therefore becomes reward-competitive
    # with navigating.  Dividing by the scenario diameter shrinks both that bonus
    # and the terminal shaping impulse to a fraction of the goal reward.
    potential_scale: float = 8.0     # = 2 * circle_radius
    r_comfort: float = 0.5           # personal-space radius (surface distance, m)
    ttc_threshold: float = 3.0       # tau_min (s)
    ttc_epsilon: float = 0.20        # eps guarding 1/tau

    # --- observation normalisation constants --------------------------------
    pos_scale: float = 6.0
    vel_scale: float = 1.5

    # --- open-loop crowd pool (throughput) ----------------------------------
    # In the invisible-robot setting the humans never see the robot, so the
    # whole crowd trajectory is a pure function of the reset seed.  Re-solving
    # ORCA inside the RL loop recomputes trajectories that are already known.
    # Pointing this at a pool file replays them instead: identical dynamics,
    # ~10x cheaper per step.  None => solve ORCA live (required if the robot
    # is visible).
    crowd_pool_path: str = None

    # --- mechanism control: env-side action low-pass filter -----------------
    # a_eff_t = (1 - alpha) * a_t + alpha * a_eff_{t-1}.  0.0 = off (default).
    # This supplies TEMPORAL ACTION COHERENCE from the DYNAMICS instead of from
    # the R5 reward term.  It is the causal control for the jerk finding: if
    # filtering rescues a no-jerk arm without changing its reward at all, then
    # R5's contribution is an optimisation/exploration effect, not a preference
    # the agent has to be paid to hold.
    action_filter_alpha: float = 0.0

    # --- instrumentation (off during training for throughput) ---------------
    emit_reward_parts: bool = False
    record_trajectory: bool = False

    # --- which reward components are ACTIVE --------------------------------
    use_goal: bool = True
    use_dist: bool = True
    use_space: bool = True
    use_ttc: bool = True
    use_jerk: bool = True


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #

def point_to_segment_dist(x1, y1, x2, y2, px, py):
    """Shortest distance from (px, py) to segment (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return math.hypot(px - x1, py - y1)
    u = ((px - x1) * dx + (py - y1) * dy) / denom
    u = min(1.0, max(0.0, u))
    return math.hypot(px - (x1 + u * dx), py - (y1 + u * dy))


def _seg_dist_to_origin(a, b):
    """Vectorised point_to_segment_dist with p = origin. a, b: (n, 2)."""
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    u = np.where(denom < 1e-12, 0.0,
                 -np.einsum("ij,ij->i", a, ab) / np.where(denom < 1e-12, 1.0, denom))
    u = np.clip(u, 0.0, 1.0)
    return np.linalg.norm(a + u[:, None] * ab, axis=1)


def _ttc_vec(rel_pos, rel_vel, comb_r):
    """Vectorised time_to_collision. Returns inf where there is no impact."""
    a = np.einsum("ij,ij->i", rel_vel, rel_vel)
    b = 2.0 * np.einsum("ij,ij->i", rel_pos, rel_vel)
    cc = np.einsum("ij,ij->i", rel_pos, rel_pos) - comb_r ** 2
    out = np.full(a.shape, np.inf)
    ok = a >= 1e-9
    disc = b * b - 4.0 * a * cc
    hit = ok & (disc > 0.0)
    if np.any(hit):
        t = (-b[hit] - np.sqrt(disc[hit])) / (2.0 * a[hit])
        out[hit] = np.where(t > 0.0, t, np.inf)
    out[ok & (cc <= 0.0)] = 0.0                      # already overlapping
    return out


def time_to_collision(rel_pos, rel_vel, combined_radius):
    """Exact TTC for two discs on constant velocities. inf if never colliding."""
    a = rel_vel[0] ** 2 + rel_vel[1] ** 2
    if a < 1e-9:
        return math.inf
    b = 2.0 * (rel_pos[0] * rel_vel[0] + rel_pos[1] * rel_vel[1])
    c = rel_pos[0] ** 2 + rel_pos[1] ** 2 - combined_radius ** 2
    if c <= 0.0:
        return 0.0                      # already overlapping
    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        return math.inf
    t = (-b - math.sqrt(disc)) / (2.0 * a)
    return t if t > 0.0 else math.inf


# --------------------------------------------------------------------------- #
#  Open-loop crowd pool
# --------------------------------------------------------------------------- #
#  Only sound when cfg.robot_visible is False.  _orca_step() appends the robot
#  to a human's neighbour list ONLY under that flag, and _respawn_human_goals()
#  reads human positions alone, so with an invisible robot the crowd is a closed
#  autonomous system: its entire trajectory is fixed by the reset seed and can be
#  computed once and replayed for every arm, every seed, training and test.
_POOL_CACHE = {}


def load_crowd_pool(path):
    """Load a pool directory. `vel` is genuinely memory-mapped.

    NOTE: np.load(..., mmap_mode=...) is SILENTLY IGNORED for .npz archives --
    it returns an NpzFile and every key access materialises the whole array in
    that process.  With one 130 MB array per concurrent worker that is real
    memory and it does not scale with pool size, so the pool is stored as a
    directory of plain .npy files instead, where mmap_mode actually works and
    the OS page cache is shared between workers.
    """
    if path not in _POOL_CACHE:
        if os.path.isdir(path):
            vel = np.load(os.path.join(path, "vel.npy"), mmap_mode="r")
            pos0 = np.load(os.path.join(path, "pos0.npy"))
            radius = np.load(os.path.join(path, "radius.npy"))
            v_pref = np.load(os.path.join(path, "v_pref.npy"))
            seeds = np.load(os.path.join(path, "seeds.npy"))
        else:                                   # legacy single-file .npz
            z = np.load(path)
            vel, pos0 = z["vel"], z["pos0"]
            radius, v_pref, seeds = z["radius"], z["v_pref"], z["seeds"]
        _POOL_CACHE[path] = {
            "vel": vel, "pos0": pos0, "radius": radius, "v_pref": v_pref,
            "seeds": seeds, "n": int(seeds.shape[0]),
            "by_seed": {int(v): i for i, v in enumerate(seeds)},
        }
    return _POOL_CACHE[path]


def _simulate_crowd(cfg, seed):
    """One open-loop crowd rollout. Returns (pos0, radius, v_pref, vel[T,n,2])."""
    assert not cfg.robot_visible, "open-loop crowd pool requires an invisible robot"
    env = CrowdNavAblationEnv(_dc_replace(cfg, crowd_pool_path=None))
    env.reset(seed=int(seed))
    T, n, dt = env.max_steps, cfg.n_humans, cfg.time_step
    pos0 = env.human_pos.copy()
    vel = np.zeros((T, n, 2), dtype=np.float64)
    for t in range(T):
        v = env._orca_step()
        vel[t] = v
        env.human_pos = env.human_pos + v * dt
        env.human_vel = v
        env._respawn_human_goals()
    return pos0, env.human_radius.copy(), env.human_v_pref.copy(), vel


def _pool_worker(args):
    return _simulate_crowd(args[0], args[1])


def build_crowd_pool(cfg, seeds, path, n_workers=None, chunksize=16, verbose=True):
    """Precompute and save the crowd rollouts for `seeds`. One-time cost."""
    import multiprocessing as mp, time as _time
    seeds = [int(s) for s in seeds]
    cfg = _dc_replace(cfg, crowd_pool_path=None)
    n_workers = n_workers or (os.cpu_count() or 4)
    t0 = _time.time()
    if n_workers > 1:
        with mp.get_context("spawn").Pool(n_workers) as pool:
            out = pool.map(_pool_worker, [(cfg, s) for s in seeds], chunksize=chunksize)
    else:
        out = [_simulate_crowd(cfg, s) for s in seeds]
    pos0 = np.stack([o[0] for o in out]).astype(np.float64)
    radius = np.stack([o[1] for o in out]).astype(np.float64)
    v_pref = np.stack([o[2] for o in out]).astype(np.float64)
    vel = np.stack([o[3] for o in out]).astype(np.float64)
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "seeds.npy"), np.asarray(seeds, dtype=np.int64))
    np.save(os.path.join(path, "pos0.npy"), pos0)
    np.save(os.path.join(path, "radius.npy"), radius)
    np.save(os.path.join(path, "v_pref.npy"), v_pref)
    np.save(os.path.join(path, "vel.npy"), vel)     # the big one; mmap-able
    if verbose:
        mb = sum(os.path.getsize(os.path.join(path, f))
                 for f in os.listdir(path)) / 1e6
        print(f"crowd pool: {len(seeds):,} scenarios, {vel.shape[1]} steps, "
              f"{mb:.0f} MB, built in {_time.time()-t0:.0f}s on {n_workers} workers "
              f"-> {path}")
    return path


# --------------------------------------------------------------------------- #
#  Environment
# --------------------------------------------------------------------------- #

class CrowdNavAblationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, cfg: EnvConfig = None, render_mode=None):
        super().__init__()
        self.cfg = cfg if cfg is not None else EnvConfig()
        self.render_mode = render_mode
        c = self.cfg

        self.max_steps = int(round(c.time_limit / c.time_step))
        self.obs_dim = 5 + 7 * c.n_humans
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._rng = np.random.default_rng(0)
        self.trajectory = []
        # Defined before the first reset so _orca_step / _respawn_human_goals are
        # safe to call on a freshly constructed env (the pool builder does this).
        self._crowd_vel = None
        self._scenario_index = -1

    # ---------------------------------------------------------------- reset --
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        c = self.cfg
        rng = self._rng

        R = c.circle_radius
        # Robot: bottom of the circle, goal antipodal.
        self.robot_pos = np.array([0.0, -R], dtype=np.float64)
        self.robot_goal = np.array([0.0, R], dtype=np.float64)
        self.robot_vel = np.zeros(2, dtype=np.float64)
        self.robot_theta = math.pi / 2.0
        self.robot_radius = c.robot_radius

        # Humans on the circle with angular + radial jitter.
        n = c.n_humans
        pool = load_crowd_pool(c.crowd_pool_path) if c.crowd_pool_path else None
        if pool is not None:
            # Replay a precomputed rollout.  A seeded reset is addressed by seed
            # (deterministic -- this is what evaluation and the paired design
            # need); an unseeded reset samples the pool, which is how training
            # draws its scenario distribution.
            idx = pool["by_seed"].get(int(seed)) if seed is not None else None
            if idx is None:
                idx = int(rng.integers(pool["n"]))
            self._scenario_index = idx
            self._crowd_vel = np.asarray(pool["vel"][idx], dtype=np.float64)
            self.human_pos = np.array(pool["pos0"][idx], dtype=np.float64)
            self.human_radius = np.array(pool["radius"][idx], dtype=np.float64)
            self.human_v_pref = np.array(pool["v_pref"][idx], dtype=np.float64)
            self.human_vel = np.zeros((n, 2))
            self.human_goal = -self.human_pos.copy()
        else:
            self._scenario_index = -1
            self._crowd_vel = None
            self._spawn_humans(rng, R, n)

        self.step_count = 0
        self.prev_action = np.zeros(2, dtype=np.float64)
        self._a_filt = np.zeros(2, dtype=np.float64)
        self.prev_dist_to_goal = float(np.linalg.norm(self.robot_pos - self.robot_goal))
        self.start_pos = self.robot_pos.copy()
        self.straight_line_dist = self.prev_dist_to_goal

        # episodic metric accumulators
        self._path_length = 0.0
        self._intrusion_steps = 0
        self._intrusion_count = 0
        self._sq_accel_sum = 0.0
        self._min_sep = math.inf
        self._prev_robot_vel = np.zeros(2)
        # behavioural instrumentation -- these are the quantities the mechanism
        # analysis needs and they cost one dot product per step.
        self._act_sum = np.zeros(2)        # sum a_t
        self._act_sq_sum = np.zeros(2)     # sum a_t^2
        self._act_lag_sum = np.zeros(2)    # sum a_t * a_{t-1}
        self._act_n_lag = 0
        self._speed_sum = 0.0
        self._progress_sum = 0.0           # sum of (d_t - d_{t+1}), signed
        self._abs_progress_sum = 0.0
        self._reward_breakdown = {k: 0.0 for k in
                                  ("goal", "dist", "space", "ttc", "jerk",
                                   "collision", "step")}
        self.trajectory = [self._snapshot()] if c.record_trajectory else []

        return self._observe(), {}


    def _spawn_humans(self, rng, R, n):
        """Rejection-sampled spawn on the circle (live path; also used to build
        the crowd pool)."""
        c = self.cfg
        base_angles = rng.uniform(0.0, 2.0 * math.pi, size=n)
        self.human_pos = np.zeros((n, 2))
        self.human_goal = np.zeros((n, 2))
        self.human_vel = np.zeros((n, 2))
        self.human_radius = rng.uniform(*c.human_radius_range, size=n)
        self.human_v_pref = rng.uniform(*c.human_v_pref_range, size=n)
        for i in range(n):
            for _ in range(100):                       # rejection-sample spawns
                ang = base_angles[i] + rng.uniform(-0.5, 0.5)
                px = R * math.cos(ang) + rng.uniform(-c.position_noise, c.position_noise)
                py = R * math.sin(ang) + rng.uniform(-c.position_noise, c.position_noise)
                ok = True
                for j in range(i):
                    min_d = self.human_radius[i] + self.human_radius[j] + 0.3
                    if math.hypot(px - self.human_pos[j, 0], py - self.human_pos[j, 1]) < min_d:
                        ok = False
                        break
                if ok:
                    d_rob = math.hypot(px - self.robot_pos[0], py - self.robot_pos[1])
                    if d_rob < self.human_radius[i] + self.robot_radius + 0.4:
                        ok = False
                if ok:
                    break
            self.human_pos[i] = (px, py)
            self.human_goal[i] = (-px, -py)

    # ----------------------------------------------------------------- step --
    def step(self, action):
        c = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)

        # ---- 0. optional env-side action low-pass (mechanism control) --------
        # Applied BEFORE anything else, so the filtered action is what drives the
        # dynamics AND what the jerk term (if active) sees.  alpha = 0 is a
        # no-op and is asserted bit-identical to the unfiltered env in section 4.
        if c.action_filter_alpha > 0.0:
            self._a_filt = ((1.0 - c.action_filter_alpha) * action
                            + c.action_filter_alpha * self._a_filt)
            action = self._a_filt

        # ---- 1. humans act first (ORCA), robot optionally visible to them ----
        human_new_vel = self._orca_step()

        # ---- 2. robot action -> velocity -------------------------------------
        if c.kinematics == "holonomic":
            v = action * c.v_pref
            if c.action_frame == "goal_aligned":
                # Actions live in the SAME goal-aligned egocentric frame as the
                # observation: a = (forward-to-goal, left-of-goal-bearing).
                dgv = self.robot_goal - self.robot_pos
                ang = math.atan2(dgv[1], dgv[0])
                ca, sa = math.cos(ang), math.sin(ang)
                v = np.array([ca * v[0] - sa * v[1], sa * v[0] + ca * v[1]])
            speed = float(np.linalg.norm(v))
            if speed > c.v_pref:
                v = v / speed * c.v_pref
            new_theta = math.atan2(v[1], v[0]) if speed > 1e-6 else self.robot_theta
        else:  # unicycle
            lin = (action[0] + 1.0) * 0.5 * c.v_pref
            ang = action[1] * c.omega_max
            new_theta = self.robot_theta + ang * c.time_step
            v = np.array([lin * math.cos(new_theta), lin * math.sin(new_theta)])
        robot_new_vel = v

        # ---- 3. collision / separation over the whole timestep --------------
        dt = c.time_step
        rel_pos = self.human_pos - self.robot_pos                  # (n, 2)
        rel_vel = human_new_vel - robot_new_vel                    # (n, 2)
        comb_r = self.robot_radius + self.human_radius             # (n,)
        seps = _seg_dist_to_origin(rel_pos, rel_pos + rel_vel * dt) - comb_r
        collision = bool(np.any(seps < 0.0))
        min_sep_step = float(seps.min())
        self._min_sep = min(self._min_sep, min_sep_step)

        # ---- 3b. proxemic bookkeeping (ALWAYS measured, even when R3 is off) -
        d_surf_now = np.linalg.norm(rel_pos, axis=1) - comb_r      # (n,)
        n_intruding = int(np.count_nonzero(d_surf_now < c.r_comfort))
        if n_intruding > 0:
            self._intrusion_steps += 1
            self._intrusion_count += n_intruding

        # ---- 4. resolve TERMINATION BEFORE computing the reward -------------
        # The potential-based shaping term needs to know whether s' is terminal
        # (Ng et al. 1999 requires Phi(terminal) = 0), so the episode outcome
        # must be resolved first.  Note `truncated` is deliberately NOT terminal:
        # the time limit is not part of the MDP, and SB3 bootstraps the value
        # there, so zeroing the potential at a timeout would be wrong.
        next_pos = self.robot_pos + robot_new_vel * dt
        dist_next = float(np.linalg.norm(next_pos - self.robot_goal))
        success = bool(dist_next < c.goal_tolerance)
        terminated = bool(collision or success)
        truncated = bool((not terminated) and (self.step_count + 1) >= self.max_steps)

        # ---- 5. reward -------------------------------------------------------
        reward, parts = self._compute_reward(
            action, robot_new_vel, human_new_vel, collision, success,
            terminated, dist_next, rel_pos, rel_vel, comb_r, d_surf_now
        )

        # ---- 6. integrate ----------------------------------------------------
        self.robot_pos = next_pos
        self.robot_vel = robot_new_vel
        self.robot_theta = new_theta
        self.human_pos = self.human_pos + human_new_vel * dt
        self.human_vel = human_new_vel
        self._respawn_human_goals()

        self._path_length += float(np.linalg.norm(robot_new_vel)) * dt
        self._sq_accel_sum += float(
            np.sum(((robot_new_vel - self._prev_robot_vel) / dt) ** 2)
        )
        self._prev_robot_vel = robot_new_vel.copy()
        # behavioural instrumentation (cheap; used by the mechanism analysis)
        if self.step_count > 0:
            self._act_lag_sum += action * self.prev_action
            self._act_n_lag += 1
        self._act_sum += action
        self._act_sq_sum += action * action
        self._speed_sum += float(np.linalg.norm(robot_new_vel))
        self._progress_sum += (self.prev_dist_to_goal - dist_next)
        self._abs_progress_sum += abs(self.prev_dist_to_goal - dist_next)

        self.prev_action = action.copy()
        self.step_count += 1
        self.prev_dist_to_goal = dist_next

        info = {"reward_parts": parts} if c.emit_reward_parts else {}
        if terminated or truncated:
            info["episode_metrics"] = self._episode_metrics(success, collision, truncated)

        if c.record_trajectory:
            self.trajectory.append(self._snapshot())
        return self._observe(), float(reward), terminated, truncated, info

    # --------------------------------------------------------------- reward --
    def _compute_reward(self, action, robot_vel, human_vel, collision, success,
                        terminated, dist_next,
                        rel_pos=None, rel_vel=None, comb_r=None, d_surf=None):
        c = self.cfg
        parts = {"goal": 0.0, "dist": 0.0, "space": 0.0,
                 "ttc": 0.0, "jerk": 0.0, "collision": 0.0, "step": 0.0}

        # --- task-level time cost: every step, every arm, never ablated ------
        parts["step"] = c.r_step

        # --- R1: terminal goal bonus -----------------------------------------
        if success and c.use_goal:
            parts["goal"] = c.w_goal

        # --- collision penalty: part of the TASK, identical in every config ---
        if collision:
            parts["collision"] = c.r_collision

        # --- R2: distance shaping --------------------------------------------
        if c.use_dist:
            L = c.potential_scale
            phi_s = -self.prev_dist_to_goal / L
            if c.dist_mode == "pbrs":
                # Ng, Harada & Russell (1999): policy invariance holds ONLY if
                # Phi(s') = 0 for terminal s'.  Omitting this is not a harmless
                # approximation -- it silently converts the shaping term into an
                # ordinary (non-invariant) reward that leaks a pseudo goal bonus
                # and destabilises the value function at episode boundaries.
                phi_s_next = 0.0 if terminated else -dist_next / L
                parts["dist"] = c.w_dist * (c.gamma * phi_s_next - phi_s)
            else:
                # 'progress': the undiscounted per-step distance reduction used
                # by most crowd-navigation papers.  Equivalent to PBRS with a
                # shaping discount of 1.0, hence NOT policy-invariant: it really
                # does add goal-directedness rather than only accelerating it.
                parts["dist"] = c.w_dist * (self.prev_dist_to_goal - dist_next) / L

        # --- R3: personal-space (proxemic) penalty ---------------------------
        # --- R4: time-to-collision penalty -----------------------------------
        if c.use_space or c.use_ttc:
            if rel_pos is None:                       # stand-alone call (audit)
                rel_pos = self.human_pos - self.robot_pos
                rel_vel = human_vel - robot_vel
                comb_r = self.robot_radius + self.human_radius
                d_surf = np.linalg.norm(rel_pos, axis=1) - comb_r
            space_pen = 0.0
            ttc_pen = 0.0
            if c.use_space:
                near = d_surf < c.r_comfort
                if np.any(near):
                    space_pen = float(np.sum(
                        (c.r_comfort - np.clip(d_surf[near], 0.0, None)) ** 2))
            if c.use_ttc:
                taus = _ttc_vec(rel_pos, rel_vel, comb_r)
                hot = taus < c.ttc_threshold
                if np.any(hot):
                    ttc_pen = float(np.sum(
                        1.0 / (taus[hot] + c.ttc_epsilon)
                        - 1.0 / (c.ttc_threshold + c.ttc_epsilon)))
            if c.use_space:
                parts["space"] = -c.w_space * space_pen
            if c.use_ttc:
                parts["ttc"] = -c.w_ttc * ttc_pen

        # --- R5: jerk / action-smoothness penalty ----------------------------
        if c.use_jerk:
            parts["jerk"] = -c.w_jerk * float(np.sum((action - self.prev_action) ** 2))

        for k, v in parts.items():
            self._reward_breakdown[k] += v
        return sum(parts.values()), parts

    # ----------------------------------------------------------------- ORCA --
    def _orca_step(self):
        c = self.cfg
        n = c.n_humans
        if self._crowd_vel is not None:          # replay a precomputed rollout
            t = min(self.step_count, self._crowd_vel.shape[0] - 1)
            return self._crowd_vel[t]
        new_vel = np.zeros((n, 2))

        agents = [(tuple(self.human_pos[i]), tuple(self.human_vel[i]),
                   float(self.human_radius[i])) for i in range(n)]
        robot_agent = (tuple(self.robot_pos), tuple(self.robot_vel), self.robot_radius)

        for i in range(n):
            to_goal = self.human_goal[i] - self.human_pos[i]
            d = float(np.linalg.norm(to_goal))
            if d > 1e-6:
                pref = tuple(to_goal / d * min(self.human_v_pref[i], d / c.time_step))
            else:
                pref = (0.0, 0.0)

            neighbours = [agents[j] for j in range(n) if j != i]
            if c.robot_visible:
                neighbours.append(robot_agent)

            new_vel[i] = orca_velocity(
                pos=agents[i][0], vel=agents[i][1], radius=agents[i][2],
                pref_vel=pref, max_speed=float(self.human_v_pref[i]),
                neighbours=neighbours,
                time_horizon=c.orca_time_horizon, time_step=c.time_step,
                neighbour_dist=c.orca_neighbour_dist,
            )
        return new_vel

    def _respawn_human_goals(self):
        c = self.cfg
        if self._crowd_vel is not None:
            return                              # goals only feed ORCA
        for i in range(c.n_humans):
            if np.linalg.norm(self.human_pos[i] - self.human_goal[i]) < c.human_goal_tolerance:
                self.human_goal[i] = -self.human_pos[i]

    # ---------------------------------------------------------- observation --
    def _observe(self):
        """Robot-centric, goal-aligned observation (the 'rotate' transform of SARL)."""
        c = self.cfg
        dgv = self.robot_goal - self.robot_pos
        dg = float(np.linalg.norm(dgv))
        ang = math.atan2(dgv[1], dgv[0])
        ca, sa = math.cos(-ang), math.sin(-ang)
        rot = np.array([[ca, -sa], [sa, ca]])

        vr = rot @ self.robot_vel
        obs = [dg / c.pos_scale, c.v_pref / c.vel_scale, self.robot_radius,
               vr[0] / c.vel_scale, vr[1] / c.vel_scale]

        d = self.human_pos - self.robot_pos                        # (n, 2)
        da = np.linalg.norm(d, axis=1)                             # (n,)
        order = np.argsort(da, kind="stable")      # nearest first (permutation-stable)
        rel = d[order] @ rot.T
        hv = self.human_vel[order] @ rot.T
        da_s = da[order]
        hr = self.human_radius[order]
        block = np.empty((c.n_humans, 7))
        block[:, 0] = rel[:, 0] / c.pos_scale
        block[:, 1] = rel[:, 1] / c.pos_scale
        block[:, 2] = hv[:, 0] / c.vel_scale
        block[:, 3] = hv[:, 1] / c.vel_scale
        block[:, 4] = hr
        block[:, 5] = da_s / c.pos_scale
        block[:, 6] = (da_s - self.robot_radius - hr) / c.pos_scale
        return np.concatenate([np.asarray(obs), block.ravel()]).astype(np.float32)

    # ---------------------------------------------------------- bookkeeping --
    def _snapshot(self):
        return {
            "robot": self.robot_pos.copy(),
            "humans": self.human_pos.copy(),
            "hr": self.human_radius.copy(),
            "goal": self.robot_goal.copy(),
        }

    def _episode_metrics(self, success, collision, truncated):
        c = self.cfg
        T = max(self.step_count, 1)
        # lag-1 action autocorrelation, averaged over the two action dimensions.
        # This is the direct measurement of "temporal action coherence" -- the
        # quantity the jerk penalty is hypothesised to buy.
        if self._act_n_lag > 0:
            mu = self._act_sum / T
            var = self._act_sq_sum / T - mu * mu
            cov = self._act_lag_sum / self._act_n_lag - mu * mu
            with np.errstate(divide="ignore", invalid="ignore"):
                ac = np.where(var > 1e-9, cov / np.where(var > 1e-9, var, 1.0), 0.0)
            action_autocorr = float(np.mean(np.clip(ac, -1.0, 1.0)))
        else:
            action_autocorr = 0.0
        net_disp = float(np.linalg.norm(self.robot_pos - self.start_pos))
        return {
            "action_autocorr": action_autocorr,
            "mean_speed": self._speed_sum / T,
            # net displacement per metre of path walked: 1.0 = perfectly
            # directed, ~0 = walked a lot and went nowhere (the failure signature
            # of an incoherent policy).
            "displacement_efficiency": net_disp / max(self._path_length, 1e-6),
            "net_progress_per_step": self._progress_sum / T,
            "success": float(success),
            "collision": float(collision),
            "timeout": float(truncated),
            "nav_time": self.step_count * c.time_step,
            "path_length": self._path_length,
            "path_length_ratio": self._path_length / max(self.straight_line_dist, 1e-6),
            "intrusion_steps": float(self._intrusion_steps),
            "intrusion_count": float(self._intrusion_count),
            "intrusion_rate": self._intrusion_steps / T,
            "mean_sq_accel": self._sq_accel_sum / T,
            "min_separation": float(self._min_sep),
            "steps": T,
            **{f"rew_{k}": v for k, v in self._reward_breakdown.items()},
        }


# --------------------------------------------------------------------------- #
#  Ablation configuration table
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#  Ablation configuration table -- the COMPLETE 2^5 factorial lattice
# --------------------------------------------------------------------------- #
#  Leave-one-out answers "what does component i add ON TOP OF the other four?".
#  That single number is context-dependent and, as v4 showed, can be wildly
#  misleading: R5's leave-one-out effect was -0.95 success while its add-one-in
#  effect was ~0.00.  Both are true; neither alone is "the effect of R5".
#
#  Enumerating all 2^5 = 32 subsets removes the choice of context entirely.  From
#  the full lattice we can compute, EXACTLY and without any modelling assumption:
#    * the Shapley value of each component (its average marginal contribution
#      over all 2^4 = 16 contexts, weighted so the values sum to v(full)-v(none)),
#    * the Moebius / Harsanyi decomposition, i.e. every pairwise, triple, ... 
#      interaction term, which is what tells substitutes from complements,
#    * leave-one-out and add-one-in as two special cases of the same object.
#
#  Component order is fixed: (goal R1, dist R2, space R3, ttc R4, jerk R5).

COMPONENT_IDS   = ("R1", "R2", "R3", "R4", "R5")
COMPONENT_KEYS  = ("goal", "dist", "space", "ttc", "jerk")
COMPONENT_NAMES = ("goal reaching", "distance shaping", "personal space",
                   "time-to-collision", "jerk / smoothness")


def lattice_name(bits):
    """(1,1,0,0,1) -> 'R125';  (0,0,0,0,0) -> 'R0_none'."""
    act = "".join(str(i + 1) for i, b in enumerate(bits) if b)
    return f"R{act}" if act else "R0_none"


def lattice_bits(name):
    """Inverse of lattice_name for pure lattice arms; None otherwise."""
    if name == "R0_none":
        return (False,) * 5
    if not (name.startswith("R") and name[1:].isdigit()):
        return None
    on = set(int(ch) for ch in name[1:])
    return tuple((i + 1) in on for i in range(5))


# The 32 lattice arms, ordered by descending component count then by index, so
# the reference sits first and the empty arm last.
LATTICE = [tuple(bool(k >> i & 1) for i in range(5)) for k in range(32)]
LATTICE.sort(key=lambda b: (-sum(b), [not x for x in b]))

ABLATION_CONFIGS = {lattice_name(b): b for b in LATTICE}

# Two extra arms that are NOT lattice members: they keep the same component set
# but swap the FORM of R2 from potential-based (policy-invariant) shaping to the
# naive per-step progress reward used by most crowd-navigation papers.  They are
# excluded from the Shapley computation by construction and analysed separately.
ABLATION_CONFIGS["P12345_progress"] = (True,  True, True, True, True)
ABLATION_CONFIGS["P2345_progress"]  = (False, True, True, True, True)

ARM_ENV_OVERRIDES = {
    "P12345_progress": {"dist_mode": "progress"},
    "P2345_progress":  {"dist_mode": "progress"},
}

# Human-readable aliases for the arms the classical leave-one-out table reports.
# These are LABELS ONLY -- `make_cfg` resolves them to the lattice arm.
ALIASES = {
    "R12345":     "R1-5_full",
    "R2345":      "A1_no-goal",
    "R1345":      "A2_no-distance",
    "R1245":      "A3_no-space",
    "R1235":      "A4_no-ttc",
    "R1234":      "A5_no-jerk",
    "R1":         "B1_goal-only",
    "R12":        "B2_goal+dist",
    "R125":       "B3_goal+dist+jerk",
    "R123":       "B4_goal+dist+space",
    "R124":       "B5_goal+dist+ttc",
    "R1245_":     "",                       # placeholder, unused
    "R2":         "B6_dist-only",
    "R25":        "B7_dist+jerk",
    "R0_none":    "Z_task-only",
    "R1245":      "A3_no-space",
    "R12345_":    "",
    "P12345_progress": "C1_progress-dist",
    "P2345_progress":  "C2_no-goal_progress",
}
ALIASES = {k: v for k, v in ALIASES.items() if v}

# Canonical leave-one-out family, in the order the main table reports them.
LOO_ARMS = ["R12345", "R2345", "R1345", "R1245", "R1235", "R1234"]
# Canonical add-one-in family (task-only plus a single component).
AOI_ARMS = ["R0_none", "R1", "R2", "R3", "R4", "R5"]


def display_name(name):
    """'R1234' -> 'A5_no-jerk (R1234)'.  Falls back to the raw lattice name."""
    a = ALIASES.get(name)
    return f"{a} [{name}]" if a else name


_ALIAS_TO_LATTICE = {v: k for k, v in ALIASES.items()}


def resolve_arm(name):
    """Accept a lattice name ('R1234'), an alias ('A5_no-jerk') or an alias with
    the lattice name attached ('A5_no-jerk [R1234]')."""
    if name in ABLATION_CONFIGS:
        return name
    if name in _ALIAS_TO_LATTICE:
        return _ALIAS_TO_LATTICE[name]
    if "[" in name and name.endswith("]"):
        inner = name[name.index("[") + 1:-1]
        if inner in ABLATION_CONFIGS:
            return inner
    raise KeyError(f"unknown ablation arm: {name!r}")


def make_cfg(name, **overrides):
    name = resolve_arm(name)
    g, d, s, t, j = ABLATION_CONFIGS[name]
    cfg = EnvConfig(use_goal=g, use_dist=d, use_space=s, use_ttc=t, use_jerk=j)
    for k, v in {**overrides, **ARM_ENV_OVERRIDES.get(name, {})}.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------- #
#  Non-learning reference policies (reported as baselines alongside PPO)
# --------------------------------------------------------------------------- #

def _to_action(env, v_world):
    """World-frame velocity -> the env's normalised action."""
    c = env.cfg
    if c.kinematics == "holonomic" and c.action_frame == "goal_aligned":
        dgv = env.robot_goal - env.robot_pos
        ang = math.atan2(dgv[1], dgv[0])
        ca, sa = math.cos(-ang), math.sin(-ang)
        v_world = np.array([ca * v_world[0] - sa * v_world[1],
                            sa * v_world[0] + ca * v_world[1]])
    return np.clip(np.asarray(v_world) / c.v_pref, -1.0, 1.0)


def orca_robot_action(env, responsibility=1.0, safety_space=0.15):
    """ORCA planner driving the robot.

    `responsibility=1.0` because the humans cannot see the robot in the
    invisible-robot setting and will not yield.  `safety_space` inflates the
    robot radius: vanilla ORCA plans to graze at exactly zero separation, which
    discrete-time integration then turns into a collision (the same trick is
    used by the ORCA baseline in Chen et al.'s CrowdNav)."""
    c = env.cfg
    to_goal = env.robot_goal - env.robot_pos
    d = float(np.linalg.norm(to_goal))
    pref = tuple(to_goal / d * min(c.v_pref, d / c.time_step)) if d > 1e-6 else (0.0, 0.0)
    neigh = [(tuple(env.human_pos[i]), tuple(env.human_vel[i]), float(env.human_radius[i]))
             for i in range(c.n_humans)]
    v = orca_velocity(pos=tuple(env.robot_pos), vel=tuple(env.robot_vel),
                      radius=env.robot_radius + safety_space,
                      pref_vel=pref, max_speed=c.v_pref,
                      neighbours=neigh, time_horizon=c.orca_time_horizon,
                      time_step=c.time_step, neighbour_dist=c.orca_neighbour_dist,
                      responsibility=responsibility)
    return _to_action(env, np.asarray(v))


def straight_line_action(env):
    return _to_action(env, (env.robot_goal - env.robot_pos) /
                      max(float(np.linalg.norm(env.robot_goal - env.robot_pos)), 1e-6)
                      * env.cfg.v_pref)


def run_baseline(policy_fn, cfg, seeds):
    """Roll out a state-based (non-neural) policy over a fixed seed list."""
    import pandas as pd
    env = CrowdNavAblationEnv(cfg)
    recs = []
    for s in seeds:
        env.reset(seed=int(s))
        ret = 0.0
        while True:
            _, r, term, trunc, info = env.step(policy_fn(env))
            ret += r
            if term or trunc:
                rec = dict(info["episode_metrics"])
                rec["scenario_seed"] = int(s)
                rec["ep_return"] = ret
                recs.append(rec)
                break
    return pd.DataFrame(recs)