"""M3-A 集成测试: 验证 FaultInjector 的真实物理故障注入。

运行 (CPU-only):
    source scripts/activate_env.sh
    export CUDA_VISIBLE_DEVICES=2
    python -m sim.test_faults            # 自包含 runner
    # 或
    pytest -q sim/test_faults.py         # 若环境有 pytest

覆盖:
- 每条故障 FaultRecord.injected 是否真实核实 (读回物理状态)。
- 故障是否真实改变物理状态 (cube 位姿/速度/关节角变化, 掉落)。
- 确定性种子: 同 seed 复现相同注入参数与最终物理状态。
"""
from __future__ import annotations

import numpy as np

from sim import PickCubeAdapter, PickCubeSkill
from sim.fault_injection import (
    FAULT_GRASP_PERTURB,
    FAULT_GRIPPER_DROP,
    FAULT_OBJECT_DISPLACEMENT,
    FAULT_OBJECT_IMPULSE,
    FaultInjector,
)

_RECORDS: dict = {}


def _run_episode(adapter, seed, injector, on_step=None, max_steps=200):
    """跑一集脚本化 pick 技能，on_step(obs, step, injector) 每步回调用于注入故障。"""
    obs = adapter.reset(seed=seed)
    skill = PickCubeSkill(adapter)
    skill.reset(obs)
    steps = 0
    for _ in range(max_steps):
        if skill.is_done:
            break
        action = skill.act(obs)
        res = adapter.step(action)
        obs = res.observation
        steps += 1
        if on_step is not None:
            on_step(obs, steps, injector)
        injector.after_step(obs)
        if res.episode.terminated or res.episode.truncated:
            break
    return obs, steps


# --------------------------------------------------------------------- #
# 测试用例
# --------------------------------------------------------------------- #
def test_object_displacement():
    adapter = PickCubeAdapter(max_episode_steps=200)
    inj = FaultInjector(adapter, seed=0)
    disp_step = [20]

    def on_step(obs, step, injector):
        if step == disp_step[0]:
            injector.inject(FAULT_OBJECT_DISPLACEMENT, step=step,
                            params={"delta": [0.08, 0.0, 0.0]})

    obs, steps = _run_episode(adapter, 0, inj, on_step)
    adapter.close()
    rec = inj.records[0]
    assert rec.fault_type == FAULT_OBJECT_DISPLACEMENT
    assert rec.step == 20
    assert rec.injected is True
    assert rec.detail["moved"] > 0.07
    assert not obs.truth.success, "位移后脚本技能应抓空失败"
    # 最终 cube 应停留在位移后的位置
    final_x = inj.cube_pose()[0]
    assert abs(final_x - rec.detail["after_pos"][0]) < 1e-3
    _RECORDS["object_displacement"] = rec


def test_object_impulse():
    adapter = PickCubeAdapter(max_episode_steps=200)
    inj = FaultInjector(adapter, seed=1)
    step_mark = [15]

    def on_step(obs, step, injector):
        if step == step_mark[0]:
            injector.inject(FAULT_OBJECT_IMPULSE, step=step,
                            params={"velocity": [0.5, 0.0, 0.0]})

    _run_episode(adapter, 0, inj, on_step)
    adapter.close()
    rec = inj.records[0]
    assert rec.fault_type == FAULT_OBJECT_IMPULSE
    assert rec.injected is True
    assert rec.detail["dv"] > 0.4, "速度冲量应真实写入 cube 线速度"
    _RECORDS["object_impulse"] = rec


def test_grasp_perturb():
    adapter = PickCubeAdapter(max_episode_steps=200)
    inj = FaultInjector(adapter, seed=2)
    step_mark = [30]

    def on_step(obs, step, injector):
        if step == step_mark[0]:
            injector.inject(FAULT_GRASP_PERTURB, step=step,
                            params={"scale": 0.25, "joints": [0, 1, 3]})

    obs, _ = _run_episode(adapter, 0, inj, on_step)
    adapter.close()
    rec = inj.records[0]
    assert rec.fault_type == FAULT_GRASP_PERTURB
    assert rec.injected is True
    assert rec.detail["qpos_delta_norm"] > 1e-3
    assert not obs.truth.success, "抓取姿态扰动后应抓空失败"
    _RECORDS["grasp_perturb"] = rec


def test_gripper_drop():
    adapter = PickCubeAdapter(max_episode_steps=200)
    inj = FaultInjector(adapter, seed=3)
    dropped = [False]

    def on_step(obs, step, injector):
        if not dropped[0] and injector.is_grasped() and injector.cube_airborne(0.05):
            dropped[0] = True
            injector.inject(FAULT_GRIPPER_DROP, step=step, duration=6)

    obs, _ = _run_episode(adapter, 0, inj, on_step)
    adapter.close()
    assert dropped[0], "应抓到 cube 并抬离桌面后触发掉落"
    rec = inj.records[0]
    assert rec.fault_type == FAULT_GRIPPER_DROP
    assert rec.injected is True
    assert rec.detail["fell"] is True or rec.detail["is_grasped_after"] is False
    assert not obs.truth.success, "掉落注入后应失败"
    _RECORDS["gripper_drop"] = rec


def test_determinism():
    """同 (env seed, fault seed) 两次运行 → 相同注入参数与相同最终物理状态。"""

    def run_once(env_seed, fault_seed):
        adapter = PickCubeAdapter(max_episode_steps=200)
        inj = FaultInjector(adapter, seed=fault_seed)
        step_mark = [20]

        def on_step(obs, step, injector):
            if step == step_mark[0]:
                injector.inject(FAULT_OBJECT_DISPLACEMENT, step=step,
                                params={"distance": 0.07})

        obs, _ = _run_episode(adapter, env_seed, inj, on_step)
        final_pose = inj.cube_pose().copy()
        adapter.close()
        rec = inj.records[0]
        return rec, final_pose

    rec1, pose1 = run_once(0, 7)
    rec2, pose2 = run_once(0, 7)
    assert rec1.params["delta"] == rec2.params["delta"], "注入参数应逐位复现"
    assert rec1.injected == rec2.injected == True  # noqa: E712
    assert np.allclose(pose1, pose2, atol=1e-6), "最终物理状态应复现"
    _RECORDS["determinism"] = rec1


# --------------------------------------------------------------------- #
# 自包含 runner (无 pytest 也可直接跑)
# --------------------------------------------------------------------- #
def main():
    tests = [
        ("object_displacement", test_object_displacement),
        ("object_impulse", test_object_impulse),
        ("grasp_perturb", test_grasp_perturb),
        ("gripper_drop", test_gripper_drop),
        ("determinism", test_determinism),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            rec = _RECORDS.get(name)
            # 打印关键证据
            print(f"[PASS] {name}")
            if name == "object_displacement":
                print(f"       delta={rec.params['delta']} moved={rec.detail['moved']:.4f} "
                      f"injected={rec.injected}")
            elif name == "object_impulse":
                print(f"       velocity={rec.params['velocity']} dv={rec.detail['dv']:.4f} "
                      f"injected={rec.injected}")
            elif name == "grasp_perturb":
                print(f"       noise={[round(n,4) for n in rec.params['noise']]} "
                      f"qpos_delta={rec.detail['qpos_delta_norm']:.4f} injected={rec.injected}")
            elif name == "gripper_drop":
                print(f"       start_z={rec.detail['start_z']:.4f} end_z={rec.detail['end_z']:.4f} "
                      f"fell={rec.detail['fell']} injected={rec.injected}")
            elif name == "determinism":
                print(f"       delta={rec.params['delta']} (复现一致)")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"[ERROR] {name}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
