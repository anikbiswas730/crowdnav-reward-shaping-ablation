"""Environment sanity checks: lattice, reward isolation, shaping, determinism."""
import numpy as np

from crowd_env import (ABLATION_CONFIGS, COMPONENT_KEYS, LATTICE, CrowdNavAblationEnv,
                       lattice_bits, lattice_name, make_cfg)

EXP = dict(n_humans=5, circle_radius=4.0, robot_visible=False)


def _run(cfg, seed, actions):
    env = CrowdNavAblationEnv(cfg)
    obs, _ = env.reset(seed=seed)
    traj, rewards = [obs.copy()], []
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        traj.append(obs.copy())
        rewards.append(r)
        if term or trunc:
            break
    return np.concatenate(traj), np.array(rewards), info


def test_lattice_is_complete():
    assert len(LATTICE) == 32
    assert len({lattice_name(b) for b in LATTICE}) == 32
    for b in LATTICE:
        assert lattice_bits(lattice_name(b)) == b


def test_each_arm_emits_only_its_terms():
    rng = np.random.default_rng(0)
    for name, bits in ABLATION_CONFIGS.items():
        env = CrowdNavAblationEnv(make_cfg(name, emit_reward_parts=True, **EXP))
        env.reset(seed=7)
        seen = set()
        for a in rng.uniform(-1, 1, size=(80, 2)):
            _, _, term, trunc, info = env.step(a)
            seen |= {k for k, v in info["reward_parts"].items() if abs(v) > 1e-12}
            if term or trunc:
                break
        expected = {k for k, on in zip(COMPONENT_KEYS, bits) if on}
        assert seen - {"collision", "step"} <= expected, name


def test_determinism():
    acts = np.random.default_rng(1).uniform(-1, 1, size=(100, 2))
    cfg = make_cfg("R12345", **EXP)
    a, ra, _ = _run(cfg, 42, acts)
    b, rb, _ = _run(cfg, 42, acts)
    assert np.array_equal(a, b) and np.array_equal(ra, rb)


def test_action_filter_off_is_a_no_op():
    acts = np.random.default_rng(2).uniform(-1, 1, size=(100, 2))
    a, ra, _ = _run(make_cfg("R12345", **EXP), 11, acts)
    b, rb, _ = _run(make_cfg("R12345", action_filter_alpha=0.0, **EXP), 11, acts)
    assert np.array_equal(a, b) and np.array_equal(ra, rb)


def test_pbrs_telescopes_on_terminal_episodes():
    rng = np.random.default_rng(3)
    checked = 0
    for s in range(120):
        cfg = make_cfg("R12", emit_reward_parts=True, **EXP)
        env = CrowdNavAblationEnv(cfg)
        env.reset(seed=5000 + s)
        phi0 = np.linalg.norm(env.robot_pos - env.robot_goal) / cfg.potential_scale
        total, t = 0.0, 0
        while True:
            _, _, term, trunc, info = env.step(rng.uniform(-1, 1, 2))
            total += cfg.gamma ** t * info["reward_parts"]["dist"]
            t += 1
            if term or trunc:
                break
        if term:
            assert abs(total - cfg.w_dist * phi0) < 1e-9
            checked += 1
    assert checked > 0
