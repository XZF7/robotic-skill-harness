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
