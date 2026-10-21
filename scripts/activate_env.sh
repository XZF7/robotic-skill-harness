#!/usr/bin/env bash
# ManiSkill 环境激活脚本
# 修复：系统 LD_LIBRARY_PATH 含 /usr/local/cuda-12.1/lib64，
# 其 libnvJitLink.so.12 (12.1.105) 会遮蔽 torch 自带的 12.4.127 版，
# 导致 libcusparse 报 undefined symbol __nvJitLinkComplete_12_4。
# 用法：source scripts/activate_env.sh
# 机器特定 conda env 路径：默认取环境变量 MANISKILL_ENV，未设置则回退到项目约定路径。
# 公开复现时请 `export MANISKILL_ENV=/path/to/your/maniskill-env` 指向你自己的环境。
ENV="${MANISKILL_ENV:-/data02/xzf/envs/maniskill}"
NVJT="$ENV/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$NVJT:${LD_LIBRARY_PATH}"
echo "[maniskill] activated (python=$ENV/bin/python)"
