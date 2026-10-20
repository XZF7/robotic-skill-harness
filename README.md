# Robotic Skill Harness

可复现的机器人操作（manipulation）系统，基于 ManiSkill + Franka Panda，面向机器人/具身智能实习作品集。

## 概览

- 仿真：ManiSkill3（SAPIEN 物理 + warp GPU 仿真）
- 机械臂：Franka Panda
- 任务：PickCube / StackCube 起步，逐步扩展
- 架构：四角色多代理协作（详见 `docs/ARCHITECTURE.md`）
- 共享接口契约：`docs/INTERFACES.md`
- 项目状态与进度：`docs/PROJECT_STATUS.md`

## 文档

| 文档 | 内容 |
|---|---|
| `docs/PROJECT_SPEC.md` | 最高项目规格（原始 spec） |
| `docs/ARCHITECTURE.md` | 架构与四角色职责 |
| `docs/INTERFACES.md` | 共享接口契约（RobotObservation 等） |
| `docs/PROJECT_STATUS.md` | 里程碑进度、已验证命令、限制 |

## 环境

- conda env: `/data02/xzf/envs/maniskill`（Python 3.10）
- 见 `docs/PROJECT_STATUS.md` 的 Milestone 0 章节

## 里程碑

M0 环境与仓库 → M1 操作基线 → M2 运行时 → M3 失败检测与恢复 → M4 学习扩展 → M5 自动基准 → M6 作品集文档
