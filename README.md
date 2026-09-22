# Reward-shaping ablation for crowd navigation

Most reward ablations in crowd navigation drop one term at a time and report the change. That number depends on which other terms happen to be present, and it can't see two terms that cover for each other. So instead of leave-one-out, this repo trains **every combination** of five reward terms (the full 2⁵ = 32 configurations, 6 seeds each) and computes exact Shapley values and pairwise interactions from the complete table.

## Main results

| | |
|---|---|
| Full reward | **0.983** success, 0.020 collision, 0 timeouts (6/6 seeds) |
| Best ORCA baseline | 0.714 success |
| Training | 192 runs × 500k steps, PPO, 500 shared held-out test scenarios |

Shapley attribution on success rate:

| Term | Shapley φ [95% CI] | Leave-one-out | Add-one-in |
|---|---|---|---|
| R1 goal bonus | +0.007 [0.002, 0.013] | +0.008 | 0.000 |
| R2 distance (PBRS) | +0.400 [0.209, 0.572] | +0.818 | 0.248 |
| R3 personal space | −0.001 [−0.079, 0.104] | +0.172 | 0.000 |
| R4 time-to-collision | −0.066 [−0.151, −0.006] | +0.004 | 0.000 |
| R5 jerk | +0.643 [0.463, 0.837] | +0.983 | 0.486 |

What stands out:

- **Leave-one-out misattributes.** It overstates R2 by 2× and gives R3 +0.172 when its average contribution is zero (that number comes from one failed seed out of six).
- **R2 and R5 are complements** (I = +0.483): 0.248 and 0.486 alone, 0.967 together. **R4 and R5 are substitutes** (I = −0.109).
- **The jerk term isn't really a comfort term.** Under Gaussian action sampling it penalises the policy's variance directly, so it anneals exploration. Final policy σ separates perfectly on whether R5 is present (0.316 ± 0.018 vs 0.433 ± 0.036, no overlap across 32 configs), and the weight sweep is non-monotone (peak at w = 0.02).
- **Reliability tracks R5 too.** No configuration without it solves more than 2 of 6 seeds.
- **R4 is the only term that helps both proxemic metrics** (+0.079 m min separation, fewer intrusions), at a small cost in success.

## Layout

```
notebooks/reward_ablation.ipynb   full pipeline, executed, with all outputs
src/                              the same modules the notebook writes out
  orca.py                         pure-Python ORCA (checked against RVO2)
  crowd_env.py                    Gymnasium env, modular reward, 2^5 lattice
  rl_utils.py                     PPO training, evaluation, budget scheduler
  ablation_stats.py               Shapley, Möbius, interactions, permutation tests, TOST
  viz.py                          figures
tests/                            unit tests (attribution maths + env checks)
results/tables/                   per-run and aggregated CSVs
results/figures/                  all generated figures
paper/                            LaTeX source (IEEE conference format)
```

## Setup

Circle crossing with 5 ORCA pedestrians on a 4 m radius, Δt = 0.25 s, 25 s horizon. The robot is invisible to the pedestrians, so it can't rely on them to get out of the way (a robot standing still is still hit in 27% of scenarios). Observation is 40-D, robot-centric and goal-aligned; action is a 2-D velocity in the same frame.

Every configuration keeps the same task reward (collision −1.25, time cost −0.005/step) and toggles:

| Term | Form | Weight |
|---|---|---|
| R1 | +w on reaching the goal | 1.0 |
| R2 | potential-based, w(γΦ(s′) − Φ(s)), Φ = −dist/8 | 0.6 |
| R3 | −w Σ max(0, 0.5 − dᵢ)² | 0.2 |
| R4 | bounded inverse time-to-collision below 3 s | 0.02 |
| R5 | −w ‖aₜ − aₜ₋₁‖² | 0.02 |

PPO (Stable-Baselines3): lr 3e-4 with linear decay, 4096-step rollouts, batch 512, 10 epochs, MLP [128, 128], `ent_coef = 0`. The entropy bonus is off on purpose: otherwise the jerk term would be the only thing opposing it, and dropping R5 would really be testing entropy regularisation.

## Reproducing

The notebook is self-contained. It writes out the same modules found in `src/`, builds the crowd pool, validates the environment, trains, evaluates and produces every table and figure.

**Kaggle** (what the results were produced on): import `notebooks/reward_ablation.ipynb`, enable internet, run all. The full run takes about 8.6 h. Training is CPU-bound, so the accelerator doesn't matter much. Set `QUICK_TEST = True` first for a ~15 min smoke run of every cell.

**Locally:**

```bash
pip install -r requirements.txt
python -m pytest tests          # 13 tests, a few seconds
jupyter notebook notebooks/reward_ablation.ipynb
```

A few things that make the run fit in one session:

- With the robot invisible, the crowd's trajectory depends only on the reset seed, so ORCA is solved once per scenario and replayed. This is checked to be bit-identical to the live simulation before training.
- Runs are parallelised at the run level (one process per run, `DummyVecEnv` inside), with BLAS threads pinned to 1.
- A budget scheduler sizes every stage from measured throughput and skips stages that won't finish. Training is resumable: finished runs are skipped on re-run.

## Statistics

Each quantity is computed exactly within each seed, and the seeds are the sample. Intervals are bootstrapped over seeds; p-values come from exact sign-flip permutation tests (all 2⁶ sign assignments). With 6 seeds the smallest two-sided p is 0.031, so interactions are classified by whether the CI excludes zero. "No effect" claims are backed by TOST equivalence tests at ±0.02 success (R1 and R4 pass; R3 doesn't, so it's reported as unresolved rather than inert).

## Limitations

Simulation only, with perfect state. ORCA pedestrians have no intent or group behaviour, so the social conclusions are about proxemic geometry, not human comfort. Weights are fixed and only switched on/off. One scenario family, one density. The action low-pass filter control for the jerk mechanism (implemented in `crowd_env.py`, `action_filter_alpha`) didn't fit in the compute budget and hasn't been run yet.

## License

MIT
