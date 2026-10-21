# Robotic Skill Harness — M5 基准协议 (Benchmark Protocol)

> 冻结点: commit `b67937d` (tag `v0.4.0`)。本协议在运行最终实验前冻结，不得事后修改评估标准。

## 1. 目的

M5 目标不是让学习模型超越规则基线，而是严谨评估现有系统、调查**离线-在线性能鸿沟**、产出可信可复现的实验证据。

当前已知结果 (M4):
- B0 无恢复 task_success=0.233；B1 规则恢复 0.700；B2 学习预判 0.167。
- 学习预测器离线 PR-AUC=0.877 > 规则检测器 0.618，但闭环操作下显著更差。

## 2. 环境 (冻结)

- python 3.10.22；torch 2.5.1+cu124；mani_skill 3.0.1；sapien 3.0.3；numpy 1.26.4；mplib 0.1.1；gymnasium 1.4.0。
- CPU-only headless，`CUDA_VISIBLE_DEVICES=0`（示例值，须非空；不占 GPU）。conda env `maniskill`。
- 复现命令: `source scripts/activate_env.sh && export CUDA_VISIBLE_DEVICES=0   # 0 为示例，换成可访问的 GPU id`（`0` 为示例，换成可访问的 GPU id）。

## 3. 被测系统

| 变体 | 描述 |
|---|---|
| B0 | 无恢复的原始操作基线 (TaskRunner) |
| B1 | 确定性规则式恢复 (RecoveryRunner + FailureDetector, max_retries=2) |
| B2 | 原始学习预判守卫 (RecoveryRunner + PredictiveGuard, threshold=0.5, 单帧触发) |
| B2-T | 时序门控学习预判 (Agent B 消融, 见 §6) |

所有变体共享同一 `PickCubeSkill` (mplib 规划 + 特权位姿, `privileged_access=True`) 与同一观测。**差异仅在于恢复层 (无/规则/学习预判)**，安全预算 (max_retries=2) 一致。

## 4. 故障场景

| 故障 | 参数 | 注入步 | 可恢复性 |
|---|---|---|---|
| object_displacement | delta=[0.07,0,0] | 10 | 可恢复 (重规划) |
| grasp_perturb | scale=0.25 | 25 | 部分可恢复 (PD 吸收) |
| object_displacement_large | delta=[0.5,0,0] | 10 | 不可恢复 (规划失败→abort) |

gripper_drop (持续故障) 保持隔离，不纳入主 benchmark (固定步调度无法接入，仅 A 独立测试)。

## 5. 种子分配 (防泄漏, 两两不相交)

| 用途 | 种子 | 说明 |
|---|---|---|
| M4-A 训练 | 0–29 | 已用 (C 重映射唯一 seed) |
| M4 基准 (旧) | 0–9 | 已用，与训练重叠 (历史遗留, 见 §10) |
| M5 验证 (B2-T 调参) | 40–59 | 新，仅用于选时序门控参数 |
| **M5 held-out 测试** | **100–129** | 新，30 独立种子，仅最终评估用 |

**规则**: 不得用 held-out 测试 (100–129) 调参；时序门控参数只在验证 (40–59) 上选；选定后冻结再上 held-out。

## 6. 故障注入 eligibility 语义

| 状态 | 定义 | 判定 |
|---|---|---|
| eligible | 故障已配置 且 episode 实际到达注入步 | 需新增记录 (step 达到 schedule 步) |
| attempted | 故障调度尝试 (FaultRecord 存在) | `extract_fault(ep) is not None` |
| verified | 注入有真实物理效果 | `fault.injected == True` (读回物理状态) |
| manifested | 故障导致失败 | `failure_manifested(ep)` (env 失败 ∨ 失败事件 ∨ FAILED 状态 ∨ 恢复触发) |

报告: 全部 episode (含 trivial-success 与未触发故障) + fault-eligible 子集 (按预定义 eligibility)。

## 7. 指标 (显式分母)

**主指标**: task_success_rate (num_success/num_episodes)；paired task-success difference (方法间配对差异)。

**次指标**: recovery_success_rate、verified_fault_injection_rate、false_recovery_trigger_rate、avg_retries、abort_rate、cumulative_reward、total_episode_steps、wall-clock_time、fault_detection_latency、early_warning_lead_time。

- false_recovery_trigger_rate 分母 = 注入且 manifested 的失败场景；分子 = 预判触发但 matched B1 首次尝试即成功 (无需恢复) 的 episode 数。
- 全部指标附 raw counts + denominators + 百分比。

**不确定性**: 成功率 95% CI (Wilson)；方法间差异用 paired bootstrap CI (按 seed 配对重采样, 反映 episode 共享 seed/fault 的分组依赖)。不得无证据声称统计显著。

## 8. 统计方法

- 配对设计: 同一 (fault, seed) 下 B0/B1/B2/B2-T 各跑一次, 差异在配对水平计算。
- paired bootstrap: 按 seed 重采样, 1000 次, 取 2.5%/97.5% 分位为 95% CI。
- 报告: 主指标差异 + CI + bootstrap 分布, 不下 "显著" 结论除非 CI 不含 0 且有支持。

## 9. 复现命令 (最终实验后补全)

```bash
source scripts/activate_env.sh && export CUDA_VISIBLE_DEVICES=0   # 0 为示例，换成可访问的 GPU id
python scripts/verify_m5.py --faults object_displacement grasp_perturb object_displacement_large \
    --seeds $(seq 100 129) --methods b0 b1 b2 --out logs/m5_primary.json
# B2-T 消融 (验证种子 40-59 选参, held-out 100-129 冻结评估)
python scripts/verify_m5.py --faults ... --seeds $(seq 100 129) --methods b2t --out logs/m5_b2t.json
```

## 10. 已知限制

- M4 基准 (seeds 0-9) 与训练 (seeds 0-29) 重叠, 属历史遗留; M5 held-out (100-129) 彻底避开。
- 特征仅 proprioception (22 维 obs.policy), 不含物体位姿 (obs.truth 禁区); 抓空失败在闭合后才可观测, 预判存在先天头启动限制。
- val 集过小 (17 episode), 阈值调优不可靠, threshold=0.5 为稳健默认。
- B2 的 num_real_failures(27) > B1(23): 预判中止本身制造失败, 佐证其有害性。
