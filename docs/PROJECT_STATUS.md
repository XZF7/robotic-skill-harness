# Project Status

> 维护规则：每个里程碑记录实际执行命令、验证结果、限制、集成依赖。未验证前不得标记完成。

## Milestone 0 — 环境与仓库 ✅（已验证）

- OS: Ubuntu 20.04.6 LTS（lthpc）；4× H800 80GB（被其他任务占用）；nvcc 12.1.105。
- 环境 `/data02/xzf/envs/maniskill`（Python 3.10.22）：torch 2.5.1+cu124、mani_skill 3.0.1、sapien 3.0.3、mplib 0.1.1、pytest。
- 官方示例 PickCube-v1（CPU headless）跑通，日志 `logs/m0_pickcube_cpu.log`。
- 详见 git 历史 `de2db2b`。

## Milestone 1 — 操作基线 + 运行时 + 评估框架 ✅（已验证并集成）

### 共享契约冻结

`interfaces/` 6 契约 + 12 契约测试（tag `v0.1.0-interfaces`）。三分支均未改动契约（已核验 git diff 为空）。

### 三 Agent 并行成果（各自 worktree/分支，已合并 master）

| Agent | 目录 | 内容 | 测试 |
|---|---|---|---|
| A 仿真 | `sim/` | PickCubeAdapter(SimulatorAdapter) + PickCubeSkill(mplib 运动规划) | 10 |
| B 运行时 | `runtime/` | SkillRegistry / SkillExecutor / TaskStateMachine / ExecutionLogger | 20 |
| D 评估 | `eval/` | 序列化 / 指标 / 基准配置 / 基线对比 / mock 守卫 | 56 |

### 集成验证（Lead 独立复测，非自报）

- 全量测试：`python -m pytest tests/ runtime/ eval/ -q` → **98 passed, 0 failed**（12 契约 + 10 sim + 20 runtime + 56 eval）。
- PickCube 基线独立重跑：`python -m sim.run_episodes -n 5 --seeds 0 1 2 3 4` → **5/5 成功（100%），平均 68.4 步**，逐 seed 71/71/47/81/72。日志 `logs/m1_pickcube_baseline.log`。
- 测试数更正：Agent A 自报"22 单测"实际含 12 个契约测试，其新增 adapter 测试为 10 个。

### 环境修复（Agent A 发现，已生效）

1. **numpy 2.2.6 → 1.26.4**：numpy2 使 mplib `ArticulatedModel` 空指针段错误。已降级，mani_skill/mplib/torch 均正常。
2. **CUDA_VISIBLE_DEVICES 不能设空串**：空串使 SAPIEN `build_cube` 段错误。规避：设为单一有效卡号（如 `2`），但仿真/渲染仍走 CPU（`-b cpu`，`render_backend=none`），不占 GPU 计算。

### M2 集成点（待办，不阻塞 M1）

- `PickCubeSkill` 扩展了接口外的 `finalize()/status/result()`；`SkillExecutor` 目前靠 `is_done` + `truth.success` 判定。二者联调（A 真实技能 + B executor）时需统一 SkillResult 的单一来源。
- Agent C 尚未启动（符合约束 §4）。

## 后续里程碑

M2 可靠运行时（A+B 联调、失败恢复）→ M3 失败检测与恢复 → M4 学习扩展（启动 Agent C）→ M5 自动基准 → M6 文档。


## Milestone 2 — 可靠技能执行运行时 ✅（已验证）

### M2-A 契约收敛（Lead）

- `RobotSkill` 新增抽象方法 `result(final_obs=None)`：技能单一权威结果（幂等，只返回 SUCCESS/FAILURE）。
- `SkillExecutor` 改为循环结束后调用 `skill.result()`，**不再读 `truth.success`**；TIMEOUT/ABORTED/异常仍由 executor 产生。
- 契约 docstring 明确区分四概念：技能完成 / 技能成功 / 任务成功 / 终止与截断。
- 特权真值 `truth` 隔离：仅评估层与脚本化技能可用，视觉策略不得访问。
- tag `v0.2.0-interfaces`。

### M2-B 真实集成（2 子代理，顺序执行——B 依赖 A 的适配结果）

- Agent A：`PickCubeSkill` 适配 `result(final_obs)`，删除 `finalize()`，保留 mplib 基线（5/5 成功）。
- Agent B：`runtime/harness.py` TaskRunner 全链路驱动 + `last_episode` 暴露 env 真值 + 7 集成测试。
- 注意：Agent B 给 `SkillExecutor` 加了 `last_episode`（加性、非破坏，经审查）。

### M2-C 端到端验证（Lead 独立复测）

- 验证脚本 `scripts/verify_m2c.py`（可复现），日志 `logs/m2c_verification.log`。
- 路径 `Task → SkillExecutor → PickCubeSkill → ManiSkill → SkillResult → Logger` 跑通。
- 3 固定种子经 harness 全部 success，`skill_status=success ↔ env_success=True` 一致（consistent=True）。
- 单一权威 SkillResult（`len(skill_results)==1`）；JSON trace 含完整字段。
- Agent D 消费真实 episode：`source_counts={'sim':3}`，`require_sim` 守卫通过。
- 全量测试 **108 passed, 0 failed**（契约 13 + sim 10 + runtime 29 + eval 56）。

### 已知小问题（不阻塞，记入后续）

- `TaskRunner.to_episode_result()` 的 `reward` 取最后一步 reward 而非累计和（当前不影响 M2 验收，M3/M5 评估前需修正）。


## Milestone 3 — 失败检测、恢复与故障注入 ✅（已验证）

### M3-0 准备（Lead）

- 修复 `to_episode_result()` reward 累计（原只取末步；现 `EpisodeResult.reward` 用累计，末步存 metrics.final_step_reward）。
- 特权真值审计：`PickCubeSkill` 在 `_plan`(物体/目标位姿)、`act`/`result`(`truth.success`) 三处用特权信息 → 显式标注 `privileged_access=True`（oracle/状态基线），四类信息隔离写入契约 docstring。
- 契约冻结 `v0.3.0-contracts`：`RecoveryAction/RecoveryDecision/RecoveryResult` + `FaultRecord` + `FailureType.INVALID_OBS/RECOVERY_EXHAUSTED` + `RobotSkill.privileged_access`。

### M3-A 真实故障注入（Agent A）

`sim/fault_injection.py` FaultInjector，4 种真实物理故障（`injected` 靠注入后读回物理状态核实，非凭意图）：object_displacement / object_impulse / grasp_perturb / gripper_drop。确定性种子。5 测试。

### M3-B 失败检测 + 恢复（Agent B）

`runtime/failure_detector.py`（只用 obs.policy 检测 INVALID_OBS/COLLISION/GRIP_FAIL/DROP，绝不读 truth）+ `runtime/recovery.py`（RecoveryManager 规则式有界重试 max_retries=2、RecoveryRunner 同 episode 内编排、每次尝试一个权威 SkillResult、结构化 trace）。30 测试。

### M3-D 恢复评估（Agent D）

`eval/fault_injection.py`（compute_recovery_metrics，8 指标显式定义分母）+ `eval/recovery_benchmark.py`（compare_b0_b1 matched 对比）。40 测试。

### M3-C 端到端验证（Lead，`scripts/verify_m3c.py`）

路径 FaultInjector → FaultInjectingAdapter → TaskRunner[B0]/RecoveryRunner[B1] → EpisodeResult(source=sim) → eval。

| 故障场景 | B0 成功 | B1 成功 | 恢复成功 | 安全中止 |
|---|---|---|---|---|
| object_displacement (0.07m) | 1/10 | 10/10 | 9/10 | 0 |
| grasp_perturb (0.25rad) | 5/10 | 10/10 | 5/10 | 0 |
| object_displacement_large (0.5m) | 1/10 | 1/10 | 0 | 9/10 |

- B0 task_success=0.233 → B1 task_success=0.700（Δ+0.467）。
- B1 检测率=1.000、恢复成功率=0.609（14/23 注入且生效的失败场景恢复）、平均重试=1.39、恢复延迟≈43 步。
- 说明：seed 8 是 trivial-success 种子（方块初始即落目标位，任务 0 步成功、故障未触发），10 种子中 9 个有效。

### 已知限制 / 技术债

- gripper_drop 需条件触发（is_grasped ∧ cube_airborne），harness 固定步调度未接入，仅在 A 的独立测试验证。
- `EpisodeResult.failures` 只含最终结果的 failure，中间尝试的失败类别在 RecoveryResult.decisions 里（评估的 failure_category_counts 不含中间失败）。
- FailureDetector 阈值（collision_qvel_threshold 等）尚未用真实动力学标定。


## Milestone 4 — 可学习失败预测与预判恢复 ✅（已验证，含诚实负面结果）

### M4-0 学习契约（Lead，`v0.4.0-contracts`）

`interfaces/learning.py`：22 维 proprioception 特征（qpos/qvel/gripper/tcp_pose，仅 `obs.policy`，严禁 `obs.truth`）、失败标注 horizon H=20（失败 episode 最后 H 步帧标 1）、数据集按 episode+seed 分组防泄漏、`FailurePredictor/FailurePrediction` 接口。13 测试（含 3 泄漏防护）。

### M4-A 数据采集（Agent A）

`sim/collect_dataset.py` 可复现采集管线（B0 式故障注入、无恢复）。150 episode 平衡数据集（76 成功/74 失败，11423 帧），4 场景（normal 60 / object_displacement 30 / grasp_perturb 30 / object_impulse 30），注入核实 90/90。剔除退化种子（cube 生成在 goal 处=1 步平凡成功、mplib 规划失败=0 步）。确定性复现 bit 级一致。

### M4-D 评估协议（Agent D）

`eval/learning_split.py`（episode 级 train/val/test 切分、seed 原子性、held-out 扰动配置）、`eval/learning_metrics.py`（precision/recall/F1/PR-AUC/FP/FN/latency/lead time，手写平均精度，分母显式）、`eval/learning_baselines.py`（majority/规则/学习三基线统一 FailurePredictor）。38 测试。

### M4-C 模型训练（Agent C）

`learning/` 包：MLP(22→32→1，769 参数，加权 BCE pos_weight=8.61，归一化仅 train 集)。held-out object_impulse（47 episode）上帧级 PR-AUC=0.877 > 规则基线 0.618 > majority 0.214；threshold=0.5 时 F1=0.772/P=0.664/FP=0.122 优于规则（F1=0.707/P=0.549/FP=0.213）。checkpoint 存 `data/m4/checkpoints/`（gitignored）。

### M4-B 运行时集成（Agent B）

`runtime/predictive.py` PredictiveGuard（逐步预测，prob≥阈值且 confidence≥min_confidence 触发提前中止）+ RecoveryMode（b1=纯规则 / b2=规则+学习预判）。预判触发复用现有 SkillStatus.FAILURE+FailureType.UNKNOWN+detail.predictive，走同一 RecoveryManager.decide（max_retries 强制），只影响「是否提前重试」，不发底层指令、不绕过安全。推理异常/无效输入/低置信度回退规则。16 测试（含真实 checkpoint 集成）。

### M4 端到端验证（Lead，`scripts/verify_m4.py`）

B0/B1/B2 matched 对比（3 故障 × 10 种子，threshold=0.5）：

| 指标 | B0 | B1 | B2 |
|---|---|---|---|
| task_success_rate | 0.233 | 0.700 | 0.167 |
| recovery_success_rate | 0.000 | 0.609 | 0.074 |
| avg_retry_count | — | 1.391 | 1.926 |
| mean_episode_length | 66.5 | 50.3 | 25.7 |

**结论（诚实负面结果）**：学习预判恢复（B2）端到端劣于规则恢复（B1），甚至劣于 B0 基线。

- 帧级：学习模型确实优于规则基线（PR-AUC 0.877 vs 0.618），因利用 qvel/tcp 上下文降误报。
- 端到端：预判触发把帧级假阳（12% FP）放大成 attempt 级中止——预测器在抓取闭合阶段误报（成功/失败前兆观测不可区分），59 次触发几乎每集 2 次，打断 B1 本可成功的重规划 attempt，max_retries=2 耗尽→abort。
- 阈值敏感性：threshold=0.8 仍劣（task_success 0.400 vs B1 1.000）；threshold=0.95 中性（1.000 = B1，但预测器仅触发 2 次、退化为规则行为）。
- 根因：PickCube 的抓空失败在抓手闭合「之后」才可观测（物体位姿在 obs.truth 禁区），proprioception 无法提前区分；规则式事后检测+重规划（B1）已近最优。

### 已知限制 / 技术债

- 预判恢复在 proprioception-only 约束下对「事后才可观测」的失败无头启动优势；若要预判价值，需引入视觉/深度等额外观测（超 M4 范围）。
- 数据 seed 跨 fault_type 复用，需 episode 重映射唯一 seed（已由 C 实现并文档化）。
- val 集过小（17 episode），阈值调优不可靠，threshold=0.5 为稳健默认。
- B2 的 num_real_failures(27) > B1(23)：预判中止本身制造失败，佐证其有害。


## Milestone 5 — 稳健基准、时序决策分析与可复现性 ✅（已验证，核心为诚实负面结果）

### M5-0 冻结（Lead）

`docs/BENCHMARK_PROTOCOL.md`（冻结协议）：held-out 种子 100–129（30，避开训练 0–29）、验证 40–59、eligibility 四层、主/次指标 + Wilson CI + paired bootstrap。

### M5-A 故障/仿真审计（Agent A，`docs/fault_audit_m5.md`）

只读审计通过：注入可复现（同 seed 逐位一致）、injected 靠读回物理状态、同 episode 恢复（仅 1 次 adapter.reset）、特权访问清单（PickCubeSkill 读 cube.pose/goal/truth.success；检测/特征/守卫只读 obs.policy）、B0/B1/B2 信息等价。发现 gripper_drop injected 语义弱（已修复：`bool(fell)` + never_grasped）。

### M5-B 时序门控消融（Agent B）

4 变体（debounce K=3/5 / phase / cooldown / advisory-only），验证种子 40–44 选参，冻结 phase 门控 `FROZEN_B2T_GATE`。19 测试。

### M5-D 扩展基准（Agent D，`scripts/verify_m5.py`）

held-out 100–129 × 3 故障 × 270 episode，eligibility 四层 + 全指标 + Wilson CI + paired bootstrap + 原始 CSV/JSON + 图。发现 mplib GIL 崩溃（分块规避）。

### M5 主结果（3 故障 × 30 种子）

| 指标 | B0 | B1 | B2 | B2-T |
|---|---|---|---|---|
| task_success | 25.6% | 65.6% | 17.8% | 20.0% |
| recovery_success | 0% | 53.7% | 11.9% | 13.3% |
| abort_rate | 0% | 34.4% | 82.2% | 80.0% |

paired diff（bootstrap）: B1−B0 +0.400 CI[0.30,0.51] 显著；B2−B1 −0.478 CI[−0.58,−0.38] 显著劣；B2−B0 −0.078 CI[−0.19,0.02] 不显著；B2T−B2 +0.022 CI[0.00,0.06] 不显著。

**核心结论（诚实负面结果）**: 学习预判（B2）端到端显著劣于规则恢复（B1），时序门控（B2-T）无法修复。根因：proprioception-only 观测空间无法在抓空发生前区分成功/失败（失败仅闭合后可观测），帧级假阳（12%）放大成 attempt 级中止（每集 ~1.84 次触发），耗尽 max_retries。B1 事后检测+重规划已近最优。不推广为"学习式天生无效"——引入视觉观测可能改变结论（超范围）。

详见 `docs/M5_RESULTS.md`。
