# Project Status

> 维护规则：每个里程碑记录实际执行命令、验证结果、限制、集成依赖。未验证前不得标记完成。

## Milestone 0 — 环境与仓库 ✅ 已验证（2026-10-20）

### 环境勘察（已验证）

| 项 | 值 |
|---|---|
| OS | Ubuntu 20.04.6 LTS（`lthpc`，x86_64，内核 5.15.0-131） |
| GPU | 4× NVIDIA H800 80GB，驱动 575.57.08 / CUDA 12.9 |
| nvcc | 12.1.105（toolkit：11 / 11.7 / 12 / 12.1 / 12.4） |
| 内存 | 1 TB（可用 ~928 GB） |
| 磁盘 | `/` 剩 247G(85%)；`/data02` 剩 326G(96%) |
| conda | 24.9.2 |

### 网络（已验证）

- PyPI 可达（pypi.org / 清华 / 阿里云）。
- **HuggingFace 不可达**（SSL 错误）。M0 无影响；M4 拉 RL 数据集需注意。
- **代理限速**：走 `127.0.0.1:7890` 时清华镜像仅 ~9.75 MB/s，直连 ~53 MB/s。→ 安装依赖一律 `unset HTTP_PROXY/HTTPS_PROXY` 后直连。

### 仓库（已建）

`/data02/xzf/robotic-skill-harness/`，git 已初始化。
```
├── README.md  .gitignore
├── docs/    PROJECT_SPEC.md · ARCHITECTURE.md · INTERFACES.md · PROJECT_STATUS.md
├── interfaces/   （共享契约 Python 定义，M1 冻结实现）
├── scripts/  activate_env.sh（环境激活 + nvjitlink 修复）
└── logs/     m0_pickcube_cpu.log（M0 验证日志）
```

### 依赖环境（已装，/data02/xzf/envs/maniskill）

| 包 | 版本 |
|---|---|
| Python | 3.10.22 |
| torch | 2.5.1+cu124（CUDA 可用，已验证） |
| mani_skill | 3.0.1 |
| sapien | 3.0.3 |
| numpy | 2.2.6 |
| gymnasium | 1.4.0 |

### 关键坑与修复（务必遵守）

1. **nvjitlink 遮蔽**：系统 `LD_LIBRARY_PATH` 含 `/usr/local/cuda-12.1/lib64`，其 `libnvJitLink.so.12`(12.1.105) 遮蔽 torch 自带 12.4.127 版，导致 `libcusparse` 报 `undefined symbol __nvJitLinkComplete_12_4`。
   修复：运行前 `source scripts/activate_env.sh`（前置 pip 自带 nvjitlink 目录）。
2. **代理限速**：安装依赖前先 `unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy`，用 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。
3. **GPU 占用**：4 张 H800 均被其他任务占用（util 51–99%）。M0 用 CPU 仿真；M4 训练前需确认 GPU 空余。

### 官方示例验证（已验证 ✅）

命令：
```bash
source scripts/activate_env.sh
python -m mani_skill.examples.demo_random_action -e "PickCube-v1" -b cpu -rb none --render-mode none -n 1 -s 0
```

结果（`logs/m0_pickcube_cpu.log`，exit=0）：
- PickCube-v1（Franka Panda）加载成功
- Action space `Box(-1,1,(8,),float32)`、控制模式 `pd_joint_delta_pos`（7 关节 + 抓手）
- 50 步 episode 完整运行（truncated=True），每步返回 reward/terminated/info（含 `is_grasped`/`is_obj_placed`）
- 随机动作 success=False（符合预期——官方安装自检示例）

### 约束与决策

- 项目数据全部置于 `/data02/xzf/`（用户指定）。
- M0 完成前未启动 M1、未派任何并行编码代理（符合 spec §9）。

## 后续里程碑

M1 操作基线 → M2 运行时 → M3 失败检测与恢复 → M4 学习扩展 → M5 自动基准 → M6 文档。详见 `PROJECT_SPEC.md` §5。
