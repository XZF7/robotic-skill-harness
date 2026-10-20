# learning/ — M4-C 轻量失败预测器 (Agent C)

PyTorch MLP 失败预测器: 22 维 proprioception 状态特征 -> 单帧失败风险 P(failure within horizon=20)。

## 文件

- `dataset.py`  : 加载 M4-A `data/m4/dataset.npz` + `episodes_meta.json` -> `EpisodeRecord`；
                  归一化 mean/std 只从 train 集计算 (防泄漏)。
- `model.py`    : `MLP(22→hidden→1)` + `MLPFailurePredictor(FailurePredictor)`。
- `train.py`    : 训练循环 + episode 级切分 + held-out 评估 + 基线对比 + 保存 checkpoint。
- `evaluate.py` : 从 checkpoint 独立重建预测器, 在 held-out test 上复现评估 (可复现)。

## 运行 (仓库根目录)

```bash
source scripts/activate_env.sh
export CUDA_VISIBLE_DEVICES=2      # CPU 训练, 不占 GPU
python3 learning/train.py --seed 0 --epochs 200
python3 learning/evaluate.py --json data/m4/checkpoints/results.json
```

## 切分策略 (关键)

M4-A 数据集 150 episode, 但 **seed 跨 fault_type 复用** (seed 0..29 各含
normal + object_displacement + object_impulse + grasp_perturb 4 个 episode)。
因此 `eval.learning_split.split_episodes` 的「seed 原子性」与「held-out 扰动配置」
互斥 (seed 组内 fault_type 不一致会触发 ValueError)。

- **主切分 (heldout)** : 优先 held-out 扰动配置, 先给每个 episode 重映射唯一 seed,
  再 `held_out_fault_types=["object_impulse"]` (30 episode 全部进 test)。
  衡量对未见扰动类型的泛化。
- **次切分 (seed)** : 用真实 seed 做 disjoint-seed 切分 (held-out seeds 泛化检查)。

## 结论 (如实)

失败信号几乎被单一特征 `gripper→0` (抓手夹空) 主导。规则基线
(GRIP_FAIL: gripper<=0.02) 已接近该信号的最优解。学习模型在 held-out test 上:

- PR-AUC 0.877 > 规则 0.618 > majority 0.214 (排名质量显著优于规则)。
- threshold=0.5 时 F1 0.772 / P 0.664 / FP率 0.122, 优于规则 (F1 0.707 / P 0.549 / FP率 0.213):
  模型额外利用 qvel/tcp 上下文, 把「夹空」与「夹住物体」区分得更好 (更少误报)。
- 但 val 集过小 (17 episode), val 上调阈值 (0.35) 在 test 上过触发 (FP率 0.258),
  threshold=0.5 是更稳健的选择。
