# Demo Guide — 三个最小可执行演示

Agent A (M6) 交付的 3 个演示脚本，覆盖「正常执行 / 可恢复失败 / 不可恢复安全中止」三种路径。
全部真实跑通（CPU-only headless），控制台输出确定性、可复现的逐步骤 trace。

## 0. 前置

```bash
source scripts/activate_env.sh && export CUDA_VISIBLE_DEVICES=0
```

三个脚本都自包含（只依赖 `interfaces` / `sim` / `runtime` 三个包），固定 `seed=0`，退出码 0 表示演示正常结束。

---

## 1. `scripts/demo_normal.py` — 正常抓取（无故障）

**组件**：`PickCubeSkill` + `TaskRunner`（B0 oracle 基线，无恢复层）。

```bash
python scripts/demo_normal.py
```

**预期输出（摘要）**：

```
Demo 1 — 正常抓取 (PickCubeSkill + TaskRunner, 无故障)
[step] env reset  episode 开始 (step=0)
[step    1] gripper=1.00 is_grasped=False success=False
  ... (reach 阶段：抓手张开，gripper=1.00)
[step   36] gripper=0.51 is_grasped=False success=False   <- 抓手开始闭合
[step   37] gripper=0.46 is_grasped=True  success=False   <- 抓住 cube
  ... (move-to-goal 阶段：保持夹持，gripper=0.46)
[step   71] gripper=0.46 is_grasped=True  success=True    <- 到达目标
[RESULT] skill_status=success  steps=71  task_state=success
         env_success=True  total_reward=21.575
[SUCCESS] 正常抓取完成：reach -> grasp -> close -> move-to-goal，无故障。
```

**解读**：`gripper` 从 1.00（全开）→ 0.46（夹住 cube，因 cube 在两指之间无法完全闭合）→ 保持，
`is_grasped` 与 `success` 依次翻转 True，最终 `task_state=success`。

---

## 2. `scripts/demo_recovery.py` — 可恢复失败（object_displacement 0.07m）

**组件**：`PickCubeSkill` + `RecoveryRunner`（B1 规则恢复：`FailureDetector` + `RecoveryManager`，`max_retries=2`）。
在第 10 步把 cube 位移 0.07m，预规划轨迹抓空。

```bash
python scripts/demo_recovery.py
```

**预期输出（摘要，省略逐步 trace）**：

```
Demo 2 — 可恢复失败 (object_displacement 0.07m, B1 规则恢复)
[step] env reset (唯一一次, 恢复期间不 reset env)

[skill] attempt 1: skill.reset -> 重新规划 (reach->grasp->close->goal)，不 reset env
  ... gripper 从 1.00 闭合到 0.00 (抓空，is_grasped 始终 False) ...
[skill] attempt 2: skill.reset -> 重新规划 (reach->grasp->close->goal)，不 reset env
  ... 重新规划后 gripper 再次闭合到 0.46，is_grasped=True，最终 success=True ...

[故障注入核实]
  injected=True  type=object_displacement  step=10  distance=0.070m

[逐次尝试]
  attempt 1: status=failure  steps=74  failure=motion_fail  msg='plan exhausted without success'
             detector -> motion_fail  final_gripper=0.00  policy_anomaly=drop
  attempt 2: status=success  steps=83  failure=None
             detector -> not invoked (attempt succeeded, no failure to detect)

[恢复决策序列]
  retry_count=0  action=retry  failure_type=motion_fail  reason='retry skill within same episode'

[RESULT] attempts=2  recovered=True  aborted=False  final_status=success
[SUCCESS] 初始失败 -> 检测 -> RETRY -> 同 episode 重规划 -> 恢复成功。
```

**三阶段解读**：

1. **失败检测**：attempt 1 抓空后抓手全闭（`final_gripper=0.00`，夹住东西时会是 ~0.46），
   技能自报 `MOTION_FAIL`（"plan exhausted without success"）。
   `FailureDetector` 检测到 `motion_fail`；纯可观测启发式（`policy_anomaly`）会报 `drop`
   （抓手先进入夹持窗再跌回全闭 = 物体丢失）。二者是同一「抓空」事实的两个视角。
2. **恢复决策**：`RecoveryManager.decide()` 对 `motion_fail` 返回 `RETRY`（`retry_count=0 < max_retries=2`）。
3. **同 episode 恢复**：重试只调 `skill.reset`（重规划，读到 cube 位移后的新位姿），**不 reset env**；
   第二次尝试直接成功 → `attempts=2, recovered=True`。

> 关于「GRIP_FAIL」：检测器对技能自报的失败类型优先规范化（此处为 `motion_fail`）。
> `GRIP_FAIL` 是检测器对「抓手全闭=夹空」的可观测启发式标签，在本场景被更早触发的
> `DROP`（曾夹持后滑脱）与技能自报优先覆盖。演示如实打印检测器的真实输出，不伪造标签。

---

## 3. `scripts/demo_abort.py` — 不可恢复失败（object_displacement_large 0.5m），安全中止

**组件**：同 demo 2。第 10 步把 cube 位移 0.5m（超出可达范围），重规划失败 → 有界重试耗尽 → ABORT。

```bash
python scripts/demo_abort.py
```

**预期输出（摘要）**：

```
Demo 3 — 不可恢复失败 (object_displacement_large 0.5m)，安全中止
[故障注入核实]
  injected=True  type=object_displacement  step=10  distance=0.500m

[逐次尝试]
  attempt 1: status=failure  steps=74  failure=motion_fail  msg='plan exhausted without success'
  attempt 2: status=failure  steps=0   failure=motion_fail  msg='reach 阶段运动规划失败'
  attempt 3: status=failure  steps=0   failure=motion_fail  msg='reach 阶段运动规划失败'

[恢复决策序列]
  retry_count=0  action=retry  failure_type=motion_fail  reason='retry skill within same episode'
  retry_count=1  action=retry  failure_type=motion_fail  reason='retry skill within same episode'
  retry_count=2  action=abort  failure_type=motion_fail  reason='recovery exhausted (retry budget reached)'  <-- RECOVERY_EXHAUSTED

[RESULT] attempts=3  recovered=False  aborted=True  final_status=failure
[ABORT] 规划失败 -> 有界重试 max_retries=2 耗尽 -> RECOVERY_EXHAUSTED -> 安全中止 (未把 reset 当作恢复)。
```

**三阶段解读**：

1. **失败检测**：attempt 1 抓空（规划在故障前完成，轨迹仍指向旧位姿）→ 失败。
2. **有界重试**：attempt 2/3 重规划时 `reach 阶段运动规划失败`（cube 位移 0.5m 超出工作空间，
   `plan_screw` 返回非 Success），`steps=0`（规划阶段即失败，未执行任何动作）。
   `RecoveryManager` 仍按 `max_retries=2` 依次给 `RETRY`（`retry_count=0,1`）。
3. **安全中止**：`retry_count=2 >= max_retries=2` → `ABORT`，`reason='recovery exhausted'`，
   产出 `RECOVERY_EXHAUSTED` 语义。**全程未把 reset 当作恢复**（env 只 reset 一次，
   恢复只是同 episode 内的 `skill.reset` 重规划）。

---

## 4. 录像说明（不伪造）

- 本机 **CPU-only headless**，4× GPU 均被无关训练任务占用。实测 `render_mode="rgb_array"` 时渲染
  触发 Vulkan `ErrorDeviceLost`（GPU 满载），且本 SAPIEN 构建的 `render_backend="cpu"` 仍回退 Vulkan，
  故**本机无法安全渲染视频**，三个演示不产出 .mp4。
- **可复现的替代记录**：脚本打印的逐步骤 trace（step/gripper/is_grasped/success + 失败类别 + 恢复决策）
  是确定性的文本记录。固定 `seed=0` 重跑即得完全相同的输出——**脚本本身就是 replay**。

### 在有健康 GPU 的机器上录制（已据 ManiSkill 3.0.1 源码核实 API，本机未执行）

```bash
source scripts/activate_env.sh
export CUDA_VISIBLE_DEVICES=<空闲 GPU id>   # 不能用已满的卡
python scripts/record_demos.py --scenario normal     # -> videos/demo_normal.mp4
python scripts/record_demos.py --scenario recovery   # -> videos/demo_recovery.mp4
python scripts/record_demos.py --scenario abort      # -> videos/demo_abort.mp4
python scripts/record_demos.py --all                 # 三个都录
```

要点（已核实，非猜测）：
- `PickCubeAdapter` 现已参数化 `render_mode`（默认 `"none"` headless）；录像时传
  `render_mode="rgb_array"` + `render_backend="gpu"`（或 `"sapien_cuda:n"` 指定卡）。
- 帧抓取用 `env.unwrapped.render_rgb_array()`（返回 `(num_envs,H,W,3)`），合成视频用
  `mani_skill.utils.visualization.misc.images_to_video(images, output_dir, video_name, fps=10)`。
- `scripts/record_demos.py` 复用三个演示的**相同机器人行为**（同一 skill/故障参数/恢复策略），
  仅附加逐帧抓取，不改变控制逻辑。
- 本仓库**不提交任何伪造录像**；`videos/` 已在 .gitignore 中，录制产物不入库。
