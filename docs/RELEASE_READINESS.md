# Release-Readiness Report — Robotic Skill Harness

> 预发布审查（release candidate）。未推送、未建公开仓库、未发布 release。

## 1. Final repository status

| 项 | 值 |
|---|---|
| HEAD | `5814e92` |
| tag | `v0.6.0`（M6 完成）|
| working tree | clean（无未提交改动）|
| 测试 | 270 pytest + 5 sim fault = **275 全通过** |
| 仓库大小 | ~2 MiB（含关键证据文件，无大二进制）|
| 敏感信息 | 无凭据/密钥/令牌；绝对路径已参数化（`MANISKILL_ENV`）|

## 2. Verified commands & test results（本轮实际执行）

```bash
source scripts/activate_env.sh && export CUDA_VISIBLE_DEVICES=0
python scripts/smoke_test.py                       # -> SMOKE PASS, exit 0 (~20s)
python -m pytest tests/ runtime/ eval/ -q          # -> 270 passed
python -m sim.test_faults                          # -> 5/5 tests passed
```

- 头条指标再次独立核对原始 CSV/JSONL：B0 25.6% / B1 65.6% / B2 17.8% / B2-T 20.0%，一致。
- seed 级聚类 bootstrap：B1−B0 +0.400 [0.356,0.456] 显著、B2−B1 −0.478 显著，结论稳健。

## 3. Public release blockers

- **无阻塞项**。待办（非阻塞）：
  1. 发布需用户明确授权（本报告止于 RC）。
  2. README 里 `git clone <this-repo>` 占位符需替换为实际 URL。
  3. 演示视频（见 §4）。
- 已处理：证据文件入库（`.gitignore` 白名单 + `logs/m5_primary_*` 等已 commit）、adapter `render_mode` 参数化、CUDA 选择可配置（docstring 示例仅，无 `os.environ` 硬编码）。

## 4. Demo video availability

- **本机无视频**：4× GPU 均被无关训练任务占用，实测渲染触发 Vulkan `ErrorDeviceLost`（GPU 满载）；本 SAPIEN 构建无可用 CPU 光栅化。
- 未伪造任何录像；演示以确定性文本 trace 为准。
- **已交付可录制路径**：`scripts/record_demos.py`（复用同一机器人行为，逐帧抓取 + `images_to_video`）+ `docs/DEMO_GUIDE.md` §4 已核实 API 的流程。在健康 GPU 机器上可生成真实视频。

## 5. README quality assessment

- 覆盖面试官 1 分钟六问：问题/独立实现/原理/最强结果/学习预测为何在线失败/如何复现（新增 "At a glance" TL;DR 一节）。
- 诚实：负面结果、oracle 辅助、仅仿真、非 vision 均明确声明；数字带 CI 与分母。

## 6. Suggested GitHub repository description & topics

**Description**（<200 字符）:
> Reproducible robot-manipulation framework for fault detection, bounded recovery, and honest evaluation of learned failure prediction (ManiSkill3 + Franka Panda).

**Topics**: `robotics`, `manipulation`, `fault-tolerance`, `failure-recovery`, `simulation`, `mani-skill`, `franka-panda`, `benchmark`, `machine-learning`

## 7. Final publication checklist

- [x] 代码/文档/测试就绪，工作树 clean
- [x] 头条数字核对原始数据一致
- [x] 无敏感信息、无大文件
- [x] MIT 许可 + 第三方 attribution
- [x] 复现命令验证通过
- [ ] **用户授权发布**（gate）
- [ ] 替换 `git clone` 占位 URL
- [ ] （可选）在 GPU 机器录制 3 个演示视频
- [ ] （可选）添加 CI badge / GitHub Pages 文档
- [ ] 创建公开仓库 + push + release（需授权）
