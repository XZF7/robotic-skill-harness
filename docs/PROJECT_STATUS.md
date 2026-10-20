# Project Status

> 维护规则：每个里程碑记录实际执行命令、验证结果、限制、集成依赖。未验证前不得标记完成。

## Milestone 0 — 环境与仓库（进行中）

### 环境勘察（已验证，2026-10-20）

| 项 | 值 |
|---|---|
| OS | Ubuntu 20.04.6 LTS（`lthpc`，x86_64，内核 5.15.0-131） |
| GPU | 4× NVIDIA H800 80GB，驱动 575.57.08 / CUDA 12.9 |
| nvcc | 12.1.105（toolkit：11 / 11.7 / 12 / 12.1 / 12.4） |
| 内存 | 1 TB（可用 ~928 GB） |
| 磁盘 | `/` 剩 247G(85%)；`/data02` 剩 326G(96%) |
| Python | 3.10.22（新 env），base 3.10.15 |
| conda | 24.9.2 |

### 网络（已验证）

- PyPI 可达：pypi.org、清华、阿里云镜像均 200（代理 127.0.0.1:7890）。
- HuggingFace 不可达（SSL 错误，exit 35）。ManiSkill3 的 assets/datasets 用 HF 版本管理；M0 PickCube 示例资产已打包进 pip 包，不受影响。后续（M4 数据集）需注意此约束。

### 仓库结构（已建）

```
robotic-skill-harness/
├── README.md
├── .gitignore
├── docs/  PROJECT_SPEC.md · ARCHITECTURE.md · INTERFACES.md · PROJECT_STATUS.md
├── interfaces/   （共享契约 Python 定义，待 M1 冻结实现）
├── scripts/      （复现脚本）
├── sim/ runtime/ learning/ eval/   （待各 Agent 建立）
```

### 依赖安装（进行中）

- conda env：`/data02/xzf/envs/maniskill`（Python 3.10）✅
- torch 2.5.1（CUDA）——安装中
- mani_skill 3.x ——待装

### 官方示例验证（待执行）

- 计划命令：`python -m mani_skill.examples.demo_random_action -e "PickCube-v1" --render-mode none --sim-backend cpu`
- 结果：pending

### 关键约束与决策

- 4 张 H800 均被现有训练任务占用（util 51–99%）。M0 示例用 CPU 仿真，不占 GPU。
- 项目数据全部置于 `/data02/xzf/robotic-skill-harness`（用户指定 /data02/xzf 下新目录）。
- 尚未启动 M1 或任何并行编码代理（符合 spec §9）。

## 后续里程碑

M1–M6 见 `PROJECT_SPEC.md` §5。
