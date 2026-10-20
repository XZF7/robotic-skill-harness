"""M2-C 端到端独立验证。

路径: Task -> SkillExecutor -> PickCubeSkill -> ManiSkill -> SkillResult -> Logger
核对:
1. 真实 PickCube 任务经 TaskRunner(executor) 跑通。
2. 技能自报结果 (skill_result.status) 与 env 真值 (episode.success) 一致。
3. 单一权威 SkillResult (harness 不二次判定)。
4. Agent D 评估框架能消费 EpisodeResult(source="sim")，require_sim 守卫通过。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import io

from interfaces import SkillStatus
from runtime import ExecutionLogger, TaskRunner
from sim import PickCubeAdapter, PickCubeSkill
from eval import compute_metrics, require_sim


def main():
    seeds = [0, 1, 2]
    episodes = []
    for seed in seeds:
        adapter = PickCubeAdapter(max_episode_steps=200)
        skill = PickCubeSkill(adapter)
        buf = io.StringIO()
        logger = ExecutionLogger(stream=buf)
        runner = TaskRunner(adapter, skill, task_id="PickCube-v1", logger=logger)
        try:
            r = runner.run(seed=seed, max_steps=200)
        finally:
            runner.close()

        sr = r.skill_result
        env_success = bool(r.episode.success) if r.episode is not None else False
        # 一致性: 技能自报 SUCCESS <=> env 真值 success
        consistent = (sr.status is SkillStatus.SUCCESS) == env_success
        print(f"[seed={seed}] skill_status={sr.status.value} env_success={env_success} "
              f"task_state={r.task_state.value} terminated={r.episode.terminated} "
              f"truncated={r.episode.truncated} steps={sr.steps} consistent={consistent}")

        ep = r.to_episode_result("PickCube-v1")
        assert ep.source == "sim", "EpisodeResult 必须是 source=sim"
        assert len(ep.skill_results) == 1, "必须只有一个权威 SkillResult"
        episodes.append(ep)

        trace = buf.getvalue().strip().splitlines()[-1]
        print(f"  trace: {trace}")

    # Agent D 消费真实 episode (不掺 mock)
    m = compute_metrics(episodes)
    require_sim(episodes, context="M2-C verification")  # 应通过（全 sim）
    print(f"\n=== Agent D 消费真实 episode ===")
    print(f"success_rate={m.success_rate:.3f} mean_steps={m.mean_steps:.1f} "
          f"num_episodes={m.num_episodes} source_counts={m.extra['source_counts']}")
    assert all(not (sr.status is SkillStatus.SUCCESS) ^ ep.success for ep in episodes
               for sr in ep.skill_results), "技能结果与 env 真值不一致"
    print("\nM2-C 端到端验证通过: 真实任务经 harness 跑通, 单一权威结果与观测一致, Agent D 可消费。")


if __name__ == "__main__":
    main()
