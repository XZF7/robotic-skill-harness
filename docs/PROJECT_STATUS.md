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
