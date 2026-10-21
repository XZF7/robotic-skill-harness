# Reproducibility — 完整复现指南

Robotic Skill Harness（ManiSkill3 + Franka Panda）演示与基准的复现步骤。
覆盖「轻量冒烟」与「全量基准」两个层级，前者比后者便宜数个数量级，用于日常快速回归。

## 1. 运行时 / 硬件要求

| 项 | 要求 |
|---|---|
| OS | Linux（本项目在 `5.15.0-131-generic` 上验证） |
| Python | 3.10（验证版本 3.10.22） |
| 加速 | **CPU-only headless**，不占 GPU；`export CUDA_VISIBLE_DEVICES=0`（任一单一卡号，非空即可；仿真实际走 CPU） |
| 显示 | 无需（`render_mode="none"` / `render_backend="none"`） |

依赖版本（冻结，与 `docs/BENCHMARK_PROTOCOL.md` 一致）：

| 包 | 版本 |
|---|---|
| mani_skill | 3.0.1 |
| sapien | 3.0.3 |
| mplib | 0.1.1 |
| torch | 2.5.1+cu124 |
| numpy | 1.26.4 |
| gymnasium | 1.4.0 |

> 环境由 `scripts/activate_env.sh` 激活，脚本内已处理一个已知的
> `libnvJitLink` 符号遮蔽问题（把 torch 自带 nvjitlink 库前置到 `LD_LIBRARY_PATH`）。

## 2. 环境激活（无硬编码路径）

本仓库不硬编码任何绝对路径。激活环境统一用 `scripts/activate_env.sh`，它从以下两处取值
（优先前者），两者都未提供时**报错并给出提示**，不会静默回退到机器特定路径：

```bash
# 方式 1: 显式指定环境（推荐，最可复现）
export MANISKILL_ENV=/path/to/conda/envs/maniskill   # conda env 根目录（含 bin/python）
source scripts/activate_env.sh

# 方式 2: 先 conda activate 一个兼容环境，脚本自动取 $CONDA_PREFIX
conda activate maniskill
source scripts/activate_env.sh
```

脚本会校验 `$ENV/bin/python` 存在、打印实际 Python 版本（期望 3.10），并按需前置
torch 自带的 nvjitlink 库。> 注意：`LD_LIBRARY_PATH` 修正只对带 NVIDIA 库的 torch 环境必要；纯 CPU 安装可跳过。

## 3. 安装依赖（新机器一次性）

```bash
conda create -n maniskill python=3.10 -y
conda activate maniskill
pip install "mani_skill==3.0.1" "sapien==3.0.3" "mplib==0.1.1" \
            "torch==2.5.1" "numpy==1.26.4" "gymnasium==1.4.0"
```

（cu124 torch 需匹配 `torch==2.5.1+cu124` 及对应 `--index-url`；CPU 安装用默认 wheel 即可。）

## 4. 轻量冒烟测试（smoke test，推荐日常回归）

**成本**：1 故障 × 1 种子 × B0/B1 = 2 个 episode，约 20 秒，CPU-only。

```bash
source scripts/activate_env.sh && export CUDA_VISIBLE_DEVICES=0
python scripts/smoke_test.py
```

**预期输出（摘要）**：

```json
{
  "fault": "object_displacement", "seed": 0,
  "b0": {"success": false, "steps": 74, "status": "failure", "injected": true},
  "b1": {"success": true,  "attempts": 2, "recovered": true, "aborted": false, "injected": true},
  "passed": true
}
SMOKE PASS
```

**通过条件**：故障注入核实 `injected=True`，且 B1（规则恢复）在同一 episode 内恢复成功
（`recovered=True`）。退出码 0 = 通过。

## 5. 演示脚本（逐步骤 trace）

三个演示脚本真实跑通（退出码 0），完整解读见 `docs/DEMO_GUIDE.md`：

```bash
python scripts/demo_normal.py     # 正常抓取，无故障 -> SUCCESS
python scripts/demo_recovery.py   # object_displacement 0.07m -> 检测 -> RETRY -> 恢复成功
python scripts/demo_abort.py      # object_displacement_large 0.5m -> 重试耗尽 -> 安全 ABORT
```

## 6. 全量基准（昂贵，仅最终评估）

全量基准由 `scripts/verify_m5.py`（Agent D）驱动，远贵于 smoke：多故障 × 多种子 × B0/B1/B2/B2-T。
复现命令与协议见 `docs/BENCHMARK_PROTOCOL.md` §9（冻结 commit `b67937d`，tag `v0.4.0`）：

```bash
source scripts/activate_env.sh && export CUDA_VISIBLE_DEVICES=0
python scripts/verify_m5.py --faults object_displacement grasp_perturb object_displacement_large \
    --seeds $(seq 100 129) --methods b0 b1 b2 --out logs/m5_primary.json
```

> 关键约束：held-out 种子 `100–129` 只能用于最终评估，不得调参；时序门控参数只在验证
> 种子 `40–59` 上选择，选定后冻结。

## 7. 复现检查清单

- [ ] `source scripts/activate_env.sh` 打印 `[maniskill] activated`。
- [ ] `python -c "import mani_skill; print(mani_skill.__version__)"` → `3.0.1`。
- [ ] `python scripts/smoke_test.py` → `SMOKE PASS`，退出码 0。
- [ ] 三个 `scripts/demo_*.py` 均退出码 0，输出与本指南一致。
- [ ] （仅最终评估）`verify_m5.py` 全量结果落入 `logs/`。
