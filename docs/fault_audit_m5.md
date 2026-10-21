# M5-A 故障与仿真审计报告 (Agent A — Simulation/Fault Audit)

日期: 2026-10-21
分支: feature/a/m5-fault-audit
审计对象 (只读 cat + 1 个最小复现): sim/fault_injection.py, sim/pick_skill.py,
sim/pickcube_adapter.py, sim/collect_dataset.py, runtime/recovery.py,
runtime/failure_detector.py, runtime/predictive.py, runtime/executor.py,
runtime/harness.py, interfaces/{skill,observation,fault,failure,adapter,learning,recovery}.py,
eval/{fault_injection,recovery_benchmark,source}.py, scripts/verify_m3c.py。

结论一句话: 代码整体健康, 无崩溃级 bug; 发现 2 个语义级问题 (eligibility 层未显式记录、
gripper_drop 固定步调度的 injected 判定偏弱) 与 1 个潜在匹配脆弱性, 均已定位到具体行号与证据。

---

## 1) 故障注入可复现性 —— 通过 (实测复现)

结论: 同一 (env seed, fault seed) + 参数, 两次注入结果逐位一致。

证据 (真实执行, CPU-only, CUDA_VISIBLE_DEVICES=2):

```text
=== determinism (rng-driven delta) ===
delta1 = [-0.0494677759706974, -0.049527157098054886, 0.0]
delta2 = [-0.0494677759706974, -0.049527157098054886, 0.0]
delta equal: True
injected: True True
moved1 = 0.07000000029802322 moved2 = 0.07000000029802322
final pose equal: True
```

机制: FaultInjector.__init__ 用 `np.random.default_rng(seed)` (fault_injection.py:76);
rng 只在参数缺省时采样 (object_displacement 的角度 fault_injection.py:144,
object_impulse 的角度 :176, grasp_perturb 的 sign :206)。reset() 明确 "不改 rng" (:330-335)。
env 侧确定性由 adapter.reset(seed=...) 保证。=> 确定性成立。

## 2) 注入故障的真实物理效果 —— 通过 (injected 靠读回判定, 非意图)

结论: 四条故障的 `injected` 均在写状态后立即读回物理状态核实, 非"我调用了 set_pose"。

证据 (fault_injection.py):
- object_displacement: `after_p = self._cube_p()` 读回, `injected = bool(moved > 1e-3)`,
  moved = ||after_p - before_p|| (:156-160); detail 含 before/after_pos + verification 说明。
- object_impulse: `after_v = self._cube_lin_vel()` 读回, `injected = bool(dv > 1e-3)` (:181-185)。
- grasp_perturb: `after_q = self._robot_qpos()[:7]` 读回, `injected = bool(d > 1e-4)` (:215-219)。
- gripper_drop: `end_z = self._cube_p()[2]` + `grasped_now = self.is_grasped()` 读回,
  `injected = bool(fell or (not grasped_now))` (:247-249) —— 见第 8 节的弱判定问题。

实测复现 (显式 delta):

```text
=== explicit delta ===
injected: True moved: 0.07999999821186066
before_pos: [-0.0007480175700038671, 0.053642693907022476, 0.019999900832772255]
after_pos : [0.0792519822716713, 0.053642693907022476, 0.019999999552965164]
final cube pose: [0.07925216 0.05364232 0.01999994]
```

位移真实写入 sapien.Entity 状态, 且最终 cube 停在位移后位置 (after_pos ≈ final pose),
物理正向积分从新状态继续, 无 reset 伪恢复。

## 3) eligible / attempted / verified / manifested 四层语义 —— 部分实现 (缺 eligible 显式记录)

逐层定位:
- attempted (FaultRecord 存在): 有。eval/fault_injection.py:294 `num_injection_attempted =
  sum(1 for f in faults if f is not None)`; collect_dataset.py:332 `injection_attempted += len(recs)`。
- verified (injected=True): 有。eval/fault_injection.py:295 `num_injected = sum(... f.injected)`。
- manifested (failure_manifested): 有。eval/fault_injection.py:133-153 定义明确 (episode.success==False
  或存在 FailureEvent 或 skill_result 失败态或 recovery 被触发), 用于 is_valid_injected_failure (:156)。
- eligible (episode 到达注入步): **无显式记录**。

问题 (语义 gap, 非崩溃): FaultRecord 只在 `after_step` 在 scheduled step 触发时产出
(fault_injection.py:309-319)。若 episode 在到达注入步之前就 terminated/truncated (规划失败=0 步、
cube 生成在 goal 处的平凡成功等), 则无任何记录, 该 episode 被静默排除在 `num_injection_attempted`
分母之外, 且无法区分「已调度但未到达注入步 (eligible-but-not-attempted)」与「从未调度 (not eligible)」。
四层在实现上实际塌缩为三层: attempted(FaultRecord 存在) 隐含 eligible(到达注入步), 反向不可检测。

缓解: collect_dataset.py:61-64 用 MIN_VALID_STEPS=10 + 换种子确定性重试过滤退化 episode,
但这是采集层的兜底, 不是 benchmark 通用保证; verify_m3c.py 无此过滤。

## 4) 恢复是否始终在同一 episode —— 通过 (env 仅 reset 一次, 重试只 skill.reset)

证据 (runtime/recovery.py):
- RecoveryRunner.run 全流程仅一次 env reset: :354 `obs = self.recording.reset(seed=seed, options=options)`。
- 重试循环 (:364-422) 每次调用 `self.executor.execute(skill, obs, ...)`, 而 executor.execute 默认
  `reset_skill=True` 只调 `skill.reset(obs)` (executor.py:62-63), **从不调 adapter.reset**。
- env 已终止/截断后不再重试: :406-409 计算 `env_ended`, 传入 manager.decide, :106-112 直接 ABORT。
- 重试观测起点 = 上次 attempt 的 final_obs (:422), skill._plan 重新用当前 env 状态 (含故障后的
  物体位姿) 规划, 语义正确。
- 故障只注入一次: scheduled 在 after_step 中 `pop` (:309), 不会因重试重复注入。

=> 约束 "恢复在同一 episode 内 (只 reset 一次 env); 每次重试仅 skill.reset, 不 reset env" 严格满足。

## 5) 控制器特权访问清单 —— 逐条证据

PickCubeSkill.privileged_access = True (sim/pick_skill.py:47)。读的特权状态:
| 位置 | 代码 | 读什么 |
|---|---|---|
| _plan (pick_skill.py:96) | `get_actor_obb(u.cube)` | cube OBB (特权物体几何/位姿) |
| _plan (:98) | `u.agent.tcp.pose.to_transformation_matrix()` | 机器人 TCP 姿态 (可观测但经 unwrapped env) |
| _plan (:106) | `u.agent.build_grasp_pose(..., u.cube.pose.sp.p)` | cube 世界系位姿 (特权) |
| _plan (:108) | `u.goal_site.pose.sp.p` | goal site 位姿 (特权) |
| _plan (:110) | `u.agent.robot.get_qpos()` | 机器人关节角 (可观测) |
| act (:158) | `obs.truth.success` | 特权任务成功真值 (提前结束判定) |
| result (:207) | `final_obs.truth.success` | 特权成功真值 (补定 SUCCESS/FAILURE) |
| _hold_action (:184) | `obs.policy.qpos` | 可观测 (无特权) |

FailureDetector: **不读 obs.truth**。validate_observation 只用 obs.policy (:74-105);
_detect_collision/_detect_drop/_detect_grip_fail 只读 obs.policy.qvel/tcp_pose/gripper (:148-200)。
grep runtime/ 中生产代码无 `.truth` 访问 (仅注释/测试中出现字面量)。

PredictiveGuard: step_hook 用 `feature_fn = extract_state_features` (predictive.py:92,143),
extract_state_features 只读 obs.policy (interfaces/learning.py:33-42)。无 env/动作写权限。

extract_state_features: 只读 p.qpos/p.qvel/p.gripper/p.tcp_pose (:36-40), 22 维, 无 truth。

=> 特权隔离成立: 失败检测/特征提取/预测守卫均只读 obs.policy; 只有脚本基线技能显式声明并读 truth。

## 6) B0 / B1 / B2 信息等价性 —— 通过

三者共享: 同一 PickCubeSkill (privileged_access=True)、同一 PickCubeAdapter (obs_mode="none",
同一 RobotObservation 流)、同一 fault 调度 (FaultInjectingAdapter)、同一 RecoveryManager 默认
max_retries=2 (recovery.py:55)。
- B0 = TaskRunner (无恢复), verify_m3c.py:114-126。
- B1 = RecoveryRunner + 规则恢复 (max_retries=2 显式), verify_m3c.py:129-145。
- B2 = RecoveryRunner + PredictiveGuard (mode=b2), recovery.py:329-335; guard 只影响"是否提前重试",
  不产动作、不改 env (predictive.py:10-14), 走同一 manager.decide 与同一 max_retries 预算。
=> 除恢复层差异外, 接收等价信息 (同一技能、同一观测、同一安全预算 max_retries=2)。

## 7) 明确声明: state-based + oracle 辅助规划, 非 vision-only

- adapter obs_mode="none" (pickcube_adapter.py:66,73), policy.images=None (:107) —— 无视觉输入。
- STATE_FEATURE_NAMES = qpos(7)+qvel(7)+gripper(1)+tcp_pose(7) = 22 维纯 proprioception
  (interfaces/learning.py:24-30)。
- 控制管线用特权物体位姿做 mplib 规划 (pick_skill.py:96-108), 属 oracle/状态基线, 非观测型策略。
=> 当前系统是 state-based + oracle 辅助规划, 不是 vision-only。

## 8) gripper_drop 未纳入主 benchmark + 条件触发语义 —— 有弱判定问题

- collect_dataset.py 主 benchmark 场景仅 normal/object_displacement/grasp_perturb/object_impulse
  (build_full_schedule :152-185), **无 gripper_drop**。
- 原因 (fault_injection.py:13-14, 51-55): gripper_drop 是持续性故障, 需要"cube 被抓离桌面后"
  才有效, 必须用条件触发 (is_grasped && cube_airborne, 见 test_faults.py:123-126) + duration>1;
  固定步调度 (_expand/schedule) 无法接入这种依赖抓取状态的持续故障。

发现的问题 (语义 bug, 位于 verify_m3c.py 的用法 + fault_injection.py 的 injected 判定):
- verify_m3c.py:39 把 gripper_drop 固定调度在 step 45 (duration 6)。若 step 45 时 cube 尚未被抓取
  (规划时序波动), _finalize_sustained 的 `injected = bool(fell or (not grasped_now))`
  (fault_injection.py:249) 会因 `not grasped_now` 为 True 而报告 injected=True —— 即"打开抓手但
  从未抓过 cube"被误判为"成功注入掉落"。`not grasped_now` 分支无法区分"掉落"与"从未抓取"。
- 相比之下 oneshot 三条故障的 injected 是严格读回物理量 (位姿/速度/关节角差), gripper_drop 的
  injected 是"fell(读回) 或 not grasped(代理)", 判定偏弱。

## 潜在匹配脆弱性 (低危)

fault 只在 episode 到达 scheduled step 时触发。若未来把 fault_step 调度在"该故障导致的失败会让
episode 提前终止"的步数之后, B0 (无恢复) 与 B1 (有恢复) 可能因终止步数不同而一个触发、一个不触发,
破坏 matched 对齐。当前 verify_m3c 的 fault_step (10/25/45) 均在自然完成 (~47 步) 之前, 暂不触发。

---

## 建议修复 (仅 sim/ 我拥有; 其余需 Lead 审批)

1. [sim/fault_injection.py] gripper_drop 的 injected 判定收紧为读回物理掉落:
   建议 `injected = bool(fell)` (或 fell 且 start 时确实 grasped, 记录 grasp_before 而非仅
   finger_before), 去掉 `or not grasped_now` 的宽泛回退, 或在 detail 中显式标注
   "never_grasped" 情况。verify_m3c.py 的固定步调度应改为条件触发 (对齐 test_faults.py 模式)。
2. [建议 Lead / 评估层] 增加显式 "eligible" 记录: FaultRecord 或 EpisodeResult.metrics 增加
   `eligible` 字段 (episode 是否到达注入步), 或在 scheduled-but-not-reached 时产出一条
   `injected=False, detail={"reason":"episode ended before fault step"}` 记录, 使四层语义显式化、
   分母可审计。当前 num_injection_attempted 静默排除未到达注入步的 episode。
3. [文档] 在 BENCHMARK_PROTOCOL 中明确 gripper_drop 为"条件触发型持续故障", 不属于固定步调度的
   主 benchmark 三故障; 并注明当前系统为 state-based + oracle 辅助规划。

（本报告为只读审计; 未修改 sim/ 生产代码, 未改 interfaces/runtime/eval/learning/docs/scripts。）
