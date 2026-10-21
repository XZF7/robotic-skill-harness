# Technical Interview Q&A — Robotic Skill Harness

> 答案均指向本仓库真实实现（`interfaces/`、`runtime/`、`sim/`、`eval/`）。诚实、不夸大。

## 1. 机器人操作管线（manipulation pipeline）如何组织？

本项目管线：`TaskRunner → SkillExecutor → PickCubeSkill → PickCubeAdapter(ManiSkill) → SkillResult → (RecoveryManager) → Logger`。核心抽象是 `RobotSkill` 接口（`plan/act/result`）与 `SimulatorAdapter`（`reset/step`），把"技能逻辑"与"仿真环境"解耦。`SkillExecutor` 驱动逐步执行并产生**单一权威结果** `SkillResult`（SUCCESS/FAILURE/TIMEOUT/ABORTED），executor 不再读环境真值 `truth.success`，而是调用 `skill.result()`——这避免多处判定不一致。技能完成≠任务成功：技能可能因超时"完成"但任务失败，二者在 `result()` 里明确区分。

## 2. 状态机与技能接口如何设计？

`TaskStateMachine` 管理任务级状态（初始化→执行→恢复→终止），技能接口用抽象方法 `result(final_obs=None)` 保证幂等（同一 episode 多次调用返回缓存结果，永不返回 TIMEOUT）。关键决策是**单一权威结果**：M2 收敛前 executor 自己读 `truth.success` 判定成功，与技能的判定可能冲突；收敛后统一由技能产出结果。恢复层的 `RecoveryStateMachine` 管理 RETRY/REOBSERVE/ABORT 三态。

## 3. 运动规划用的是什么？本项目的贡献在哪？

运动规划直接用 mplib（继承，非本项目实现），物体/目标位姿在 `PickCubeSkill._plan` 里读取**特权状态**（`cube.pose`、`goal_site.pose`）并显式标注 `privileged_access=True`。本项目的贡献是**执行编排、故障注入、检测、恢复、评估管线**，不是规划算法本身——这是必须诚实区分的原创性边界。

## 4. 故障注入如何保证"真实"？

`FaultInjector` 注入 4 类物理故障（object_displacement/object_impulse/grasp_perturb/gripper_drop），注入后**读回物理状态**核实（`injected=True` 靠 `moved>1e-3`/`dv>1e-3`/`qpos_delta`/`fell` 判定，非凭意图）。同一 seed 两次注入结果逐位一致（确定性）。这保证 benchmark 里的"注入成功"是可验证的物理事实，而非调度意图。

## 5. 失败检测如何实现？为什么只读 `obs.policy`？

`FailureDetector` 只用策略观测 `obs.policy`（qpos/qvel/tcp_pose/gripper）检测 INVALID_OBS（NaN/越界）、COLLISION（qvel 突增）、GRIP_FAIL（抓手全闭夹空）、DROP（抓取后滑脱），**绝不读 `obs.truth`**——有专项测试证明 `truth.success=True` 时仍按 policy 判失败。这是防"观测泄漏"的关键：检测器与真实策略共享同一观测，不能偷看仿真真值。

## 6. 恢复策略与重试预算如何设计？

`RecoveryManager` 做确定性有界重试：`decide()` 返回 RETRY/REOBSERVE/ABORT，`max_retries=2` 强制，达上限产 `RECOVERY_EXHAUSTED→ABORT`。`RecoveryRunner` 保证恢复发生在**同一 episode 内**（仅一次 `adapter.reset`，重试只 `skill.reset`），env 终止后立即 ABORT 不重试。核心原则：**不把 reset 当恢复**——重试是重规划/重观测，不是重新开始 episode。

## 7. 重试预算耗尽如何避免"无限重试"？

有界重试 `max_retries=2` + 确定性决策：重试次数是显式预算，达上限安全中止（ABORT），不会因失败信号反复触发无限循环。M5 的可恢复故障（object_displacement 0.07m）B1 恢复 9/10；不可恢复故障（0.5m 大位移）B1 9/10 安全中止（规划失败→重试 3 次→RECOVERY_EXHAUSTED→ABORT），证明预算机制正确区分"可恢复"与"应中止"。

## 8. 仿真评估如何保证可信/可复现？

held-out 种子（100–129，避开训练 0–29）配对设计：同一 (fault, seed) 下 B0/B1/B2/B2-T 各跑一次，差异在配对水平计算。指标带显式分母与原始计数；成功率 95% Wilson CI；方法间差异用 paired bootstrap（1000 次）+ **seed 级聚类 bootstrap**（30 seed 跨 3 fault 复用，聚类反映非独立性）。原始 episode 结果落 CSV/JSON/JSONL，复现命令固化（含 mplib GIL 崩溃的分块规避）。

## 9. 观测泄漏（observation leakage）如何防范？

多层隔离：(a) 训练/评估特征只用 `obs.policy`（22 维 proprioception），物体位姿在 `obs.truth` 禁区；(b) 学习标签由完整 episode 未来结果**离线**生成，模型输入只用当前/历史观测；(c) 数据集按 episode（非帧）切分，防帧级泄漏；(d) 归一化统计量只从 train 集算；(e) held-out 扰动配置（object_impulse）整体进 test。审计确认特征/检测/守卫无 `truth` 访问。

## 10. 离线指标好为何闭环反而差？（本项目最核心的教训）

学习预测器离线 PR-AUC=0.877（帧级"20 步内失败"分类）优于规则基线 0.618，但闭环 task_success 仅 17.8%，**显著劣于规则恢复 65.6%**。原因：(a) 帧级假阳（12%）被"单帧触发即中止"放大成 attempt 级中止，每集 ~1.84 次误触发，迅速耗尽 max_retries；(b) 抓空失败在抓手闭合后才可观测，proprioception 无法提前区分成功/失败，21.3 步"提前预警"实为抓取阶段误报；(c) 干预代价不对称——中止不可逆，而规则检测只在失败**之后**介入、不浪费 attempt。教训：**帧级分类指标 ≠ 决策级有用性**，预测模型与其上的策略必须分开评估；时序门控（phase 门控）无法修复（+0.022 不显著）。这是诚实记录的负面结果，不推广为"学习式天生无效"——引入视觉观测可能改变结论。


---

## 附录：2 分钟项目走查（实习面试用）

> 技术准确、对机器人面试官友好，约 250 词 / 2 分钟口语。

**（问题）** 我在 ManiSkill 仿真里搭了一个机器人技能执行框架，研究"操作失败后如何可靠恢复"。

**（系统怎么工作）** 核心是一套分层架构：`RobotSkill` 接口把技能逻辑和仿真环境解耦，`SkillExecutor` 驱动执行并产出一个**单一权威结果**；上面叠了故障注入、失败检测、有界恢复三层。故障注入是真实物理扰动——物体被顶开、抓手被撞偏；检测器只读策略观测、绝不偷看仿真真值；恢复是有界重试，最多 2 次，够了就安全中止，绝不靠 reset 作弊。

**（我实现的）** 执行编排、故障注入、检测、恢复、评估管线都是我自己写的；运动规划用的是 mplib、仿真用 ManiSkill/SAPIEN——这部分不是我做的，我会清楚区分。

**（最强结果）** 规则式恢复把任务成功率从 25.6% 拉到 65.6%，90 个配对故障场景、bootstrap 置信区间 [0.30, 0.51]，统计上显著。

**（什么失败了，为什么）** 我还训练了一个学习式失败预测器，离线 PR-AUC 0.877 很漂亮，但接入闭环后成功率反而掉到 17.8%。原因是帧级假阳被"单帧触发即中止"放大成整次尝试的浪费，重试预算被耗尽；而且抓空失败在抓手闭合之后才可观测，本体感觉根本无法提前预测。时序门控也救不回来。

**（未来）** 要真正从失败预测里获益，得加视觉/深度观测，让模型能在失败发生前看到物体相对位姿。这个负面结果本身就是我最扎实的工程教训：离线指标不等于闭环有用性。
