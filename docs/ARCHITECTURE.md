# Architecture

## 顶层目标

可复现的机器人操作系统：ManiSkill3 + Franka Panda，四角色多代理协作开发。机器人控制与代理编排严格分离（约束 #8）。

## 模块布局（规划）

```
robotic-skill-harness/
├── docs/                # PROJECT_SPEC / ARCHITECTURE / INTERFACES / PROJECT_STATUS
├── interfaces/          # 共享契约 Python 定义（冻结后由 Lead 维护）
│   ├── observation.py   # RobotObservation
│   ├── skill.py         # RobotSkill / SkillResult
│   ├── adapter.py       # SimulatorAdapter（抽象协议）
│   ├── failure.py       # FailureEvent
│   └── episode.py       # EpisodeResult
├── sim/                 # Agent A：ManiSkill 集成 + Franka 技能 + 运动规划基线
├── runtime/             # Agent B：SkillRegistry / Executor / 状态机 / 监控 / 恢复 / 结构化日志
├── learning/            # Agent C：训练管线 + 失败预测模型（M4 启用）
├── eval/                # Agent D：自动基准 / 故障注入 / 指标
└── scripts/             # 复现脚本（一键跑示例、基准等）
```

## 四角色职责与边界

| Agent | 拥有目录 | 职责 | 对外依赖 |
|---|---|---|---|
| **A** Robotics & Simulation | `sim/` | ManiSkill 集成、Franka Panda 仿真、PickCube/StackCube、技能实现、运动规划基线 | `interfaces/adapter.py` 实现 |
| **B** Runtime Harness | `runtime/` | Skill Registry、Skill Executor、任务状态机、执行监控、失败恢复、结构化日志 | `interfaces/skill.py`、`failure.py` 实现 |
| **C** Robot Learning | `learning/` | PyTorch 训练、失败预测、数据集收集、评估、与运行时集成（M4 启用） | 依赖 A 的仿真数据 + B 的稳定接口 |
| **D** Evaluation | `eval/` | 自动基准、故障注入、episode 执行、基线对比、指标与报告 | `interfaces/episode.py` 实现 + A/B 产物 |
| **Lead** | `docs/`、`interfaces/`、`scripts/` | 架构、派发、协调、代码审查、集成、里程碑验证 | — |

## 数据流

```
ManiSkill env ──SimulatorAdapter──> 技能执行(Agent A) ──SkillResult──> Runtime 状态机(Agent B)
                                         │                                   │
                                    FailureEvent ───────────────────> 失败预测(Agent C, M4)
                                                                             │
                                   EpisodeResult <── 基准/故障注入 ── eval(Agent D)
```

## Git 分支 / worktree 约定（约束 #3）

- `main`：始终可复现的稳定分支，只经 Lead 合并。
- 每个 Agent 任务用独立 worktree + 分支：`feature/<agent>/<task>`，例如 `feature/a/pickcube-baseline`。
- 共享契约 `interfaces/` 一旦冻结，任何 Agent 不得擅自修改；变更须经 Lead 审批。
- 禁止两个 Agent 同时改同一文件（约束 #8）。

## 关键设计决策

1. **不用 LLM planner**（约束 #8）：控制由技能/状态机驱动，学习模块（Agent C）只做失败预测，不替代控制。
2. **M0 仿真用 CPU + headless**：4 张 H800 现有其他训练任务，M0 示例不占 GPU；M4 训练前再启用 GPU。
3. **共享接口冻结优先**：并行开发前先冻结 `interfaces/` 契约，防止接口漂移。
