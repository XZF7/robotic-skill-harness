# Robotic Skill Harness

A reproducible robot-manipulation framework for **fault detection, bounded recovery, and honest evaluation** of learned failure prediction — built on ManiSkill3 + SAPIEN with a Franka Panda arm.

> This is a research/engineering portfolio project (simulation only). It is **not** peer-reviewed or published research, and it is **not** a vision-based system: motion planning uses privileged simulator state (see [Known limitations](#known-limitations)).

## At a glance (one-minute summary)

- **Problem** — real manipulation is unreliable: objects get displaced, grasps slip, plans fail. A robust system needs fault *detection*, *bounded recovery*, and *honest evaluation* of "smarter" learned predictors.
- **Built here (not third-party)** — the skill/execution architecture, physical fault injection, rule-based detection & recovery, the benchmark pipeline, and the learned-predictor analysis. Motion planning (mplib), simulation (ManiSkill), and physics (SAPIEN) are third-party and used as-is.
- **How it works** — a `RobotSkill` interface (single authoritative `SkillResult`) → `SkillExecutor` → the simulator → a `RecoveryManager` with a hard `max_retries=2` budget (safe abort, never `reset`-as-recovery).
- **Strongest result** — rule-based recovery lifts task success **25.6% → 65.6%** across 90 matched fault scenarios (paired bootstrap CI [+0.30, +0.51]).
- **Negative result (honest)** — a proprioception-only MLP (offline PR-AUC 0.877) *dropped* closed-loop success to 17.8%: the failure (grasp-miss) is only observable *after* the gripper closes, so its "early warnings" were false positives that wasted retry budget.
- **Reproduce** — `source scripts/activate_env.sh && python scripts/smoke_test.py` (~20 s), then `python -m pytest tests/ runtime/ eval/` (270 tests).

## The problem

Real robot manipulation is unreliable: objects get displaced, grasps slip, plans fail. A robust system must (1) *detect* when an attempt has gone wrong, (2) *recover* with bounded retries instead of infinite loops, and (3) *evaluate* honestly whether a "smarter" learned failure predictor actually helps closed-loop performance — or just looks good on offline metrics.

## Key engineering contributions

- **Modular skill/execution architecture** — a `RobotSkill` interface with a *single authoritative* `SkillResult`, decoupling skill logic from the simulator (`SimulatorAdapter`), plus task/execution/recovery state machines.
- **Physical fault injection** — four real dynamics faults (object displacement/impulse, grasp perturbation, gripper drop) whose injection is *verified by reading back simulator state*, not by intent.
- **Rule-based failure detection & bounded recovery** — a detector that uses only policy observations (`obs.policy`, never simulator ground truth), and a recovery manager with an explicit `max_retries=2` budget that safely aborts instead of treating `reset` as recovery.
- **Learning-based failure prediction with honest offline→online analysis** — a proprioception-only MLP predictor (offline PR-AUC 0.877) that *reduced* closed-loop success, with a documented root-cause analysis of the frame-level vs. decision-level gap.
- **Reproducible matched-pair benchmark** — held-out seeds, verified-injection accounting, Wilson CIs, paired + seed-clustered bootstrap over 270 held-out episodes.

## Results at a glance (30 held-out seeds × 3 faults)

| Method | Task success | Recovery success | Abort rate |
|---|---|---|---|
| B0 — no recovery | 25.6% | — | — |
| **B1 — rule-based recovery** | **65.6%** | 53.7% (36/67) | 34.4% |
| B2 — learned predictive guard | 17.8% | 11.9% (10/84) | 82.2% |
| B2-T — phase-gated learned guard | 20.0% | 13.3% (11/83) | 80.0% |

Paired task-success difference (bootstrap): **B1−B0 = +0.400 [0.30, 0.51]** (significant); **B2−B1 = −0.478 [−0.58, −0.38]** (significant, worse). Full analysis, confidence intervals, ablations, and the offline→online gap discussion: [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md).

## Architecture

```
Task ─▶ SkillExecutor ─▶ PickCubeSkill ─▶ PickCubeAdapter ─▶ ManiSkill/SAPIEN
                                     └─▶ SkillResult ─▶ RecoveryManager ─▶ Logger
```

Complete Mermaid diagrams and component specs (interfaces, contracts, error codes, retry/logger semantics): [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Demonstrations

Three runnable demos (CPU-only, deterministic, text traces — no fabricated video):

| Demo | Command | Shows |
|---|---|---|
| Normal grasp | `python scripts/demo_normal.py` | successful PickCube execution |
| Recoverable failure | `python scripts/demo_recovery.py` | fail → detect → retry → same-episode re-plan → success |
| Unrecoverable failure | `python scripts/demo_abort.py` | plan failure → bounded retries → safe abort |

Full guide (including how to record video on a GPU/display machine): [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md).

## Installation

Requirements: Linux, Python 3.10, CPU-only is fine. The code depends on [ManiSkill3](https://maniskill.readthedocs.io/), SAPIEN, mplib, PyTorch, NumPy, gymnasium.

```bash
git clone https://github.com/XZF7/robotic-skill-harness && cd robotic-skill-harness
# create/activate your ManiSkill conda env, then point the helper at it:
export MANISKILL_ENV=/path/to/your/maniskill-env
source scripts/activate_env.sh   # sets PATH + fixes the nvjitlink LD_LIBRARY_PATH issue
```

## Quick start (smoke test, ~20 s, 2 episodes)

```bash
source scripts/activate_env.sh && export CUDA_VISIBLE_DEVICES=0
python scripts/smoke_test.py        # expects "SMOKE PASS", exit 0
```

## Benchmark reproduction

Full instructions (dependency versions, seed allocation, chunked runs to avoid the mplib GIL crash): [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

```bash
# primary B0/B1/B2 (270 episodes, held-out seeds 100–129, chunked):
for start in 100 105 110 115 120 125; do
  python scripts/verify_m5.py --tag primary     --seeds $(seq $start $((start+4)))     --faults object_displacement grasp_perturb object_displacement_large
done
# B2-T ablation + B2 vs B2-T comparison:
python scripts/compare_b2t.py --b2 logs/m5_primary_episodes.jsonl --b2t logs/m5_b2t_episodes.jsonl
```

Run the test suite: `python -m pytest tests/ runtime/ eval/ -q` (270 tests) + `python -m sim.test_faults` (5 tests).

## Repository structure

```
interfaces/   shared contracts (RobotSkill, adapter, observation, recovery, learning)
sim/          PickCube skill + adapter, fault injection, dataset collection
runtime/      executor, task runner, failure detector, recovery, predictive guard
eval/         benchmark, metrics, learning split/metrics/baselines
learning/     MLP failure predictor (model, dataset, train, evaluate)
scripts/      demos, smoke test, benchmark + comparison scripts
docs/         architecture, technical report, benchmark protocol, results, demos, ...
logs/         raw benchmark artifacts (CSV/JSON/figure; gitignored bulk)
```

## Known limitations

- **Simulation only** — ManiSkill/SAPIEN; no physical-robot validation; results do not generalize to real hardware without real-world experiments.
- **Oracle-assisted control** — `PickCubeSkill` motion planning reads privileged object/goal poses (`privileged_access=True`); this is *not* vision-only manipulation.
- **Proprioception-only learned predictor** — 22-dim state features; the object pose is intentionally excluded, which is exactly why preemptive failure prediction has no actionable head start for grasp-miss faults.
- **Restricted scope** — one task (PickCube), a small fault set, and a small training dataset (150 episodes).
- The B2 learned-predictor result is a **documented negative result**; it must not be read as "learning is inherently ineffective."

## License & attribution

This project's own code is MIT-licensed (see `LICENSE`). It builds on [ManiSkill](https://github.com/haosulab/ManiSkill), [SAPIEN](https://sapien.ucsd.edu/), and [mplib](https://github.com/haosulab/mplib) — motion planning (mplib) and simulation/physics (ManiSkill/SAPIEN) are third-party and retain their own licenses. This repository does not include those dependencies.
