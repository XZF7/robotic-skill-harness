# Resume Bullets — Robotic Skill Harness

> 每条约 25 词以内，基于真实验证结果，不声称独立实现 ManiSkill/mplib 算法。

1. **Manipulation execution architecture** — Designed a modular skill-execution framework (RobotSkill interface, task runner, skill registry, state machine) on ManiSkill3/SAPIEN for Franka Panda pick-and-place, enforcing a single authoritative `SkillResult` and clean skill-completion vs task-success semantics.

2. **Fault detection and bounded recovery** — Built physical fault injection (object displacement/impulse/grasp perturbation) and a rule-based recovery stack (failure detector + retry/abort state machine, max_retries=2) that raised task success from 25.6% to 65.6% across 90 matched fault scenarios (paired bootstrap CI [+0.30, +0.51]).

3. **Reproducible benchmark infrastructure** — Engineered an automated matched-pair evaluation pipeline with episode-level data splitting, verified fault-injection accounting, Wilson confidence intervals, and paired/bootstrap statistics over 270 held-out simulation episodes.

4. **Learning-based failure prediction + offline-to-online analysis** — Trained a proprioception-only MLP failure predictor (offline PR-AUC 0.877) and showed it *reduced* closed-loop success (17.8%), then diagnosed the frame-level-vs-decision-level gap (false-positive aborts, retry-budget exhaustion, limited observability) — a documented negative result.
