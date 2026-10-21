#!/usr/bin/env bash
# 激活 ManiSkill 环境（可移植版，无机器特定路径）。
#
# 本项目仿真走 CPU（sim_backend="cpu"、render_backend="none"），不占 GPU；
# 但 torch(cu124) 需其自带 nvjitlink 库前置到 LD_LIBRARY_PATH，否则系统 CUDA 的
# libnvJitLink.so.12 会遮蔽导致 libcusparse undefined symbol __nvJitLinkComplete_12_4。
#
# 用法（两种，任选其一；均未提供则报错，不会静默回退到任何机器特定路径）:
#   1) 显式指定环境（推荐，可复现）:
#        export MANISKILL_ENV=/path/to/your/maniskill-env
#        source scripts/activate_env.sh
#   2) 已在当前 shell 激活兼容 conda 环境（自动取 $CONDA_PREFIX）:
#        conda activate maniskill
#        source scripts/activate_env.sh

if [[ -n "${MANISKILL_ENV:-}" ]]; then
    ENV="$MANISKILL_ENV"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    ENV="$CONDA_PREFIX"
else
    echo "[maniskill] 错误: 未指定 ManiSkill 环境。" >&2
    echo "  请任选其一:" >&2
    echo "    export MANISKILL_ENV=/path/to/your/maniskill-env && source scripts/activate_env.sh" >&2
    echo "    conda activate <你的 maniskill 环境> && source scripts/activate_env.sh" >&2
    return 1 2>/dev/null || exit 1
fi

# 校验 1: python 必须存在且可执行
if [[ ! -x "$ENV/bin/python" ]]; then
    echo "[maniskill] 错误: $ENV/bin/python 不存在或不可执行。" >&2
    echo "  请检查 MANISKILL_ENV 是否指向 conda 环境根目录（应含 bin/python）。" >&2
    return 1 2>/dev/null || exit 1
fi

# 校验 2: 必须是兼容的 ManiSkill 环境（能找到 mani_skill 包）
if ! "$ENV/bin/python" -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('mani_skill') else 1)" >/dev/null 2>&1; then
    echo "[maniskill] 错误: $ENV 不是兼容的 ManiSkill 环境（找不到 mani_skill 包）。" >&2
    echo "  请确认该 conda 环境已安装 mani_skill==3.0.1（见 docs/REPRODUCIBILITY.md §3）。" >&2
    return 1 2>/dev/null || exit 1
fi

# 校验 3: Python 版本应为 3.10（项目冻结验证版本）
PYVER="$("$ENV/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo unknown)"
if [[ "$PYVER" != "3.10" ]]; then
    echo "[maniskill] 警告: 期望 Python 3.10，实际 $PYVER（依赖版本在 3.10 上冻结验证）。" >&2
fi

export PATH="$ENV/bin:$PATH"

# torch(cu124) 自带 nvjitlink 库前置，修复 libnvJitLink 符号遮蔽
NVJT="$ENV/lib/python${PYVER}/site-packages/nvidia/nvjitlink/lib"
if [[ -d "$NVJT" ]]; then
    export LD_LIBRARY_PATH="$NVJT:${LD_LIBRARY_PATH:-}"
else
    echo "[maniskill] 注意: 未找到 $NVJT（纯 CPU torch 可忽略此提示）。" >&2
fi

echo "[maniskill] activated (python=$ENV/bin/python, py=$PYVER)"
