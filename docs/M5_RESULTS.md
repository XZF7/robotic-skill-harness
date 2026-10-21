# M5 结果 — 稳健基准、时序决策分析与可复现性

> 基准协议: `docs/BENCHMARK_PROTOCOL.md`（冻结于 commit `d54d3fc`）。held-out 种子 100–129（30 独立种子，避开训练 0–29 与 M4 旧基准 0–9）。全部真实执行，非推算。

## 1. 主结果: B0/B1/B2 matched 对比（3 故障 × 30 种子 = 90 matched units, 270 episode）

| 指标 | B0 无恢复 | B1 规则恢复 | B2 学习预判 |
|---|---|---|---|
| task_success_rate | 23/90 = 25.6% | 59/90 = 65.6% | 16/90 = 17.8% |
| 95% Wilson CI | [17.7%, 35.4%] | [55.3%, 74.6%] | [11.2%, 26.9%] |
| recovery_success_rate | 0% (0/67) | 53.7% (36/67) | 11.9% (10/84) |
| abort_rate | 0% | 34.4% (31/90) | 82.2% (74/90) |
| avg_retries | — | 1.418 | 1.893 |

**paired task-success difference（paired bootstrap 1000 次）**:

| 对比 | mean | 95% CI | 结论 |
|---|---|---|---|
| B1 − B0 | +0.400 | [0.300, 0.511] | 显著优于（CI 不含 0）|
| B2 − B0 | −0.078 | [−0.189, 0.022] | 无显著差异（CI 含 0）|
| B2 − B1 | −0.478 | [−0.578, −0.378] | 显著劣于（CI 不含 0）|

**结论**: B1 规则恢复显著提升任务成功率；B2 学习预判不仅未提升，还**显著劣于 B1，且与 B0 无显著差异**（即学习预判把规则恢复的收益完全抵消）。

## 2. B2-T 时序门控消融（phase 门控, 冻结于 `FROZEN_B2T_GATE`）

Agent B 实现 4 变体（debounce K=3/5、phase-aware gating、cooldown、advisory-only），在验证种子 40–44 选参，phase 门控（gripper_closed ≤ 0.5）胜出并冻结。held-out 100–129 上:

| 指标 | B2 单帧 | B2-T phase 门控 |
|---|---|---|
| task_success | 16/90 = 17.8% | 18/90 = 20.0% |
| 95% Wilson CI | [11.2%, 26.9%] | [13.0%, 29.4%] |
| paired B2T−B2 | — | +0.022 CI[0.000, 0.056] |
| total_triggers | 166 | 161 |
| abort | 74/90 | 72/90 |
| recovery_success | 10/84 | 11/83 |

**结论**: phase 门控对 B2 几乎无改善（+0.022，CI 恰好触及 0，不显著），B2-T 仍远劣于 B1（0.200 vs 0.656）。**B2 差不是决策策略问题，时序门控无法修复**。

## 3. 离线-在线性能鸿沟的根因分析

**现象**: 学习预测器离线 PR-AUC=0.877 > 规则 0.618（帧级更优），但闭环 task_success 0.178 << 规则 0.656（端到端更差）。

**根因（4 条，均有证据）**:

1. **帧级假阳放大成 attempt 级中止**：预测器 12% 帧级假阳率；PredictiveGuard 单帧即触发；每集平均 1.84 次触发（166/90）。每次触发中止当前 attempt，不可逆。max_retries=2 下 2–3 次误触发即耗尽预算 → abort（B2 abort_rate 82.2%）。
2. **"提前预警"实为抓取阶段误报**：lead_time=21.3 步（B2 首触发 vs B1 首失败步）看似"提前"，但这 21 步是抓取接近/闭合阶段的误报——proprioception（抓手闭合）在"将夹住"与"将夹空"之间不可区分。规则检测器 B1 只在夹空**之后**才判失败，因此不浪费 attempt。
3. **观测空间先天限制**：特征仅 22 维 proprioception（obs.policy），不含物体位姿（obs.truth 禁区）。抓空失败的本质是"抓手与物体相对位置错位"，此信息在 proprioception 中不可得 → 预判无 actionable head-start。Agent A 审计确认：控制管线用特权物体位姿做 mplib 规划（`privileged_access=True`），是 state-based + oracle 辅助，非 vision-only。
4. **B1 已近最优**：对这些故障，规则式事后检测 + 重规划已恢复 36/67 真实失败（53.7%），学习预判只有误伤空间、无提升空间。

**重要限定**: 上述结论针对【proprioception-only 观测 + 事后才可观测的抓空类故障】。**不推广为"学习式方法天生无效"**——若引入视觉/深度（可提前观测物体相对位姿），预判失败可能具备真实头启动价值（超 M5 范围）。

## 4. 故障 eligibility 与注入核实

- eligible（到达注入步）270/270；attempted（FaultRecord）270/270；verified（injected=True）270/270。
- manifested（failure_manifested）：B0=67/90、B1=67/90、B2=84/90。**B2 的 manifested 比 B0/B1 多 17 个**——这 17 个是 B2 预判中止自身制造的"失败"，独立佐证其有害性（对 matched 场景，B1 无需恢复即成功，B2 却中止到失败）。
- 注入核实：Agent A 实测同 seed 两次注入 delta/moved/final pose 逐位一致（确定性）；injected 靠读回物理状态（非凭意图）。
- 修复（M5-A 审计后）：gripper_drop `injected` 由 `bool(fell or not grasped)` 收紧为 `bool(fell)` + `never_grasped` 标注（原语义无法区分"真掉落"与"从未抓取"）。

## 5. 模型评估审计（M5 §6）

| 审计项 | 结论 |
|---|---|
| episode 级切分 | ✓ `eval/learning_split.py`（seed 原子、held-out 扰动配置）|
| 未来信息泄漏 | ✓ 标签由完整 episode 未来结果离线生成，模型输入只用当前/历史观测 |
| 故障标识泄漏 | ✓ 特征为 22 维 proprioception，不含 fault_type |
| 仿真真值泄漏 | ✓ 特征只读 obs.policy，无 obs.truth |
| 归一化 | ✓ mean/std 仅 train 集计算 |
| 时间依赖 | ✓ 按 episode（非帧）切分 |
| H=20 定义 | ✓ 失败 episode 最后 20 步帧标 1，成功全 0 |
| 帧级 vs 决策级 | ✗ 帧级 PR-AUC 未转化为决策级有用性（本报告 §3 已量化）|

## 6. 结论（M5 §10）

1. **实验证明了什么**: B1 规则恢复显著有效（+0.400）；B2 学习预判显著有害（−0.478 vs B1）；B2-T 时序门控无法修复（+0.022 不显著）。
2. **仍不确定**: 引入视觉/深度观测后学习预判能否获益（未测，超范围）；gripper_drop 持续故障的完整集成。
3. **时序门控是否提升性能**: 否（phase 门控 +0.022 不显著，B2-T 仍远劣于 B1）。
4. **当前观测空间是否足够提前预测失败**: 不足——proprioception-only 无法在抓空发生前区分成功/失败，失败仅在闭合后可观测。
5. **M6 应强调**: (a) 规则式事后恢复的鲁棒性与收益（这是可靠的正面结果）；(b) 学习预判失败的诚实剖析（观测空间限制 + 帧级指标与决策级目标的错位）；(c) 作为可复现负面结果的实验价值（protocol + held-out + CI + bootstrap 全链路）。

## 7. 交付物与复现

- `docs/BENCHMARK_PROTOCOL.md`（冻结协议）、`docs/M5_RESULTS.md`（本文）、`docs/fault_audit_m5.md`（Agent A 审计）。
- 原始数据: `logs/m5_primary_episodes.csv/.json/.jsonl`（270 episode）、`logs/m5_primary.json`（聚合+CI+bootstrap）、`logs/m5_primary_figure.png`（B0/B1/B2 图）、`logs/m5_b2t_episodes.jsonl`（90 episode）、`logs/m5_b2t_comparison.json`（B2 vs B2-T）。
- 脚本: `scripts/verify_m5.py`（B0/B1/B2/B2-T 基准）、`scripts/compare_b2t.py`（B2 vs B2-T 消融）。

```bash
source scripts/activate_env.sh && export CUDA_VISIBLE_DEVICES=0
# 主基准 (分块规避 mplib GIL, 断点续跑):
for start in 100 105 110 115 120 125; do python scripts/verify_m5.py --tag primary   --seeds $(seq $start $((start+4))) --faults object_displacement grasp_perturb object_displacement_large; done
# B2-T 消融:
for start in 100 105 110 115 120 125; do python scripts/verify_m5.py --tag b2t   --seeds $(seq $start $((start+4))) --faults object_displacement grasp_perturb object_displacement_large --methods b2t --no-figs; done
# B2 vs B2-T 对比:
python scripts/compare_b2t.py --b2 logs/m5_primary_episodes.jsonl --b2t logs/m5_b2t_episodes.jsonl
```

## 8. 已知限制

- mplib/pybind11 GIL 崩溃（单进程 ~90 episode 后 segfault），须分块运行（已在复现命令固化）。
- B2 的 false_recovery_trigger_rate=25.4%（17/67），以 B1 首尝试成功为参考；预判触发的全部有害性（含 B1 也需恢复的场景）更广。
- seed 8 式 trivial-success（方块初始落目标位）在 30 种子 held-out 中未剔除，计入"全部 episode"（协议要求），另报 fault-eligible 子集。
