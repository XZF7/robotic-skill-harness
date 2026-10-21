"""运行 N 个固定种子 episode，测量 PickCube 脚本化 pick 技能的真实成功率。

用法:
    source scripts/activate_env.sh
    export CUDA_VISIBLE_DEVICES=0   # 不能为空字符串 (SAPIEN 会段错误)；单一空闲卡即可
    python -m sim.run_episodes --num-episodes 5 --seeds 0 1 2 3 4

环境前置条件 (2026-10-20 已验证):
- numpy 必须 < 2 (1.26.4)。numpy 2.2.6 会使 mplib 的 ArticulatedModel 原生段错误
  (空函数指针调用)。已降级: pip install "numpy==1.26.4" "mplib==0.1.1" --no-deps
- CUDA_VISIBLE_DEVICES 不能设为空字符串 (SAPIEN build_cube 段错误)。
  仿真/渲染仍为 CPU (-b cpu, render_backend=none)，不占用 GPU 计算。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from interfaces import EpisodeMetrics, EpisodeResult, SkillStatus
from sim import PickCubeAdapter, PickCubeSkill


def run_one_episode(adapter: PickCubeAdapter, seed: int) -> EpisodeResult:
    obs = adapter.reset(seed=seed)
    skill = PickCubeSkill(adapter)
    skill.reset(obs)

    max_steps = adapter.max_episode_steps
    steps = 0
    reward_sum = 0.0
    while steps < max_steps:
        if skill.is_done:
            break
        action = skill.act(obs)
        res = adapter.step(action)
        obs = res.observation
        reward_sum += res.reward
        steps += 1
        if res.episode.terminated or res.episode.truncated:
            break

    skill_result = skill.result(obs)
    success = bool(obs.truth.success)
    return EpisodeResult(
        task_id="PickCube-v1",
        success=success,
        steps=steps,
        reward=reward_sum,
        skill_results=[skill_result],
        failures=[skill_result.failure] if skill_result.failure else [],
        metrics={
            "seed": seed,
            "terminated": bool(skill_result.status is SkillStatus.SUCCESS),
            "skill_status": skill_result.status.value,
            "is_grasped": bool(obs.truth.is_grasped),
        },
        source="sim",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", "-n", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="固定种子列表；默认 0..num_episodes-1")
    parser.add_argument("--max-episode-steps", type=int, default=200)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(args.num_episodes))
    seeds = seeds[: args.num_episodes]

    adapter = PickCubeAdapter(max_episode_steps=args.max_episode_steps)
    results = []
    try:
        for seed in seeds:
            r = run_one_episode(adapter, seed)
            results.append(r)
            print(f"[seed={seed:3d}] success={r.success}  steps={r.steps}  "
                  f"status={r.skill_results[0].status.value}", flush=True)
    finally:
        adapter.close()

    n = len(results)
    n_success = sum(r.success for r in results)
    metrics = EpisodeMetrics(
        num_episodes=n,
        success_rate=(n_success / n) if n else 0.0,
        mean_steps=(sum(r.steps for r in results) / n) if n else 0.0,
        mean_reward=(sum(r.reward for r in results) / n) if n else 0.0,
        extra={"seeds": seeds},
    )

    print(f"\n=== 结果 ===\n成功 {n_success}/{n}  episode，成功率 {metrics.success_rate:.2%}")
    print(f"平均步数 {metrics.mean_steps:.1f}")
    # 输出 JSON 便于下游解析
    print(json.dumps({
        "num_episodes": metrics.num_episodes,
        "success_count": n_success,
        "success_rate": metrics.success_rate,
        "mean_steps": metrics.mean_steps,
        "per_episode": [asdict(r) for r in results],
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
