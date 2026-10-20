"""M4-C 训练脚本: 轻量 PyTorch MLP 失败预测器。

流程:
1. 加载 M4-A 数据集 (150 episode) -> List[EpisodeRecord]。
2. episode 级切分 (eval.learning_split.split_episodes):
   - 主切分 (heldout): object_impulse 作为 held-out 扰动配置全部进 test。
     因 M4-A 的 seed 跨 fault_type 复用 (seed 0..29 各含 normal+3 故障),
     seed 原子性与 held-out-fault-type 互斥 —— 此实验优先 held-out-fault-type,
     故先给每个 episode 重映射唯一 seed。
   - 次切分 (seed): 用真实 seed 做 disjoint-seed 切分 (held-out seeds 泛化检查)。
3. 归一化: mean/std 只从 train 集计算, 应用到 val/test (防泄漏)。
4. 确定性种子 (torch/numpy/random manual_seed), CPU 训练, 加权 BCE (pos_weight) 处理类不平衡。
5. 阈值在 val 上按 F1 扫描选定, 在 held-out test 上评估 (eval.learning_metrics),
   对比 majority + 规则式 (runtime.failure_detector) 基线。
6. 保存 checkpoint (model.pt) + config.json (含归一化统计/超参/结构)。

用法 (仓库根目录):
  python3 learning/train.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from interfaces import EpisodeRecord, STATE_FEATURE_DIM, FailurePredictor  # noqa: E402
from eval.learning_split import split_episodes  # noqa: E402
from eval.learning_metrics import (  # noqa: E402
    average_precision, compute_prediction_metrics,
)
from eval.learning_baselines import (  # noqa: E402
    MajorityClassPredictor, RuleBasedPredictor,
)
from learning.dataset import (  # noqa: E402
    load_episodes, episodes_to_arrays, compute_normalization_stats, normalize,
)
from learning.model import MLP, MLPFailurePredictor, predict_probs_batch  # noqa: E402

CHECKPOINT_DIR = os.path.join(_REPO, "data", "m4", "checkpoints")


@dataclass
class TrainConfig:
    input_dim: int = STATE_FEATURE_DIM
    hidden_dims: List[int] = field(default_factory=lambda: [32])
    dropout: float = 0.3
    lr: float = 3e-4
    weight_decay: float = 1e-3
    epochs: int = 200
    batch_size: int = 256
    seed: int = 0
    val_frac: float = 0.15
    test_frac: float = 0.15
    split_seed: int = 0
    held_out_fault_types: List[str] = field(default_factory=lambda: ["object_impulse"])
    default_threshold: float = 0.5
    device: str = "cpu"

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["hidden_dims"] = list(d["hidden_dims"])
        d["held_out_fault_types"] = list(d["held_out_fault_types"])
        return d


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 单线程保证 CPU 上逐位可复现。
    torch.set_num_threads(1)


def _unique_seed_episodes(episodes: List[EpisodeRecord]) -> List[EpisodeRecord]:
    """给每个 episode 重映射唯一 seed (全局索引), 保留其它字段。"""
    return [replace(ep, seed=i) for i, ep in enumerate(episodes)]


def _f1_at(y_true: np.ndarray, y_score: np.ndarray, t: float) -> float:
    preds = (y_score >= t).astype(np.int64)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return (2.0 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def _select_threshold(y_val: np.ndarray, score_val: np.ndarray) -> Tuple[float, float]:
    """在 val 上扫描阈值, 返回 (最佳阈值, 对应 F1)。"""
    best_t, best_f1 = 0.5, _f1_at(y_val, score_val, 0.5)
    for t in np.linspace(0.05, 0.95, 19):
        f1 = _f1_at(y_val, score_val, float(t))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def _model_metrics(test_eps: List[EpisodeRecord], predictor: FailurePredictor,
                   threshold: Optional[float]) -> dict:
    m = compute_prediction_metrics(test_eps, predictor, threshold=threshold)
    return m.to_dict()


def _baseline_metrics(test_eps: List[EpisodeRecord],
                      train_eps: List[EpisodeRecord]) -> Dict[str, dict]:
    majority = MajorityClassPredictor.fit(train_eps)
    rule = RuleBasedPredictor()
    out: Dict[str, dict] = {}
    out["majority"] = compute_prediction_metrics(test_eps, majority, threshold=None).to_dict()
    out["rule"] = compute_prediction_metrics(test_eps, rule, threshold=None).to_dict()
    return out


def train_one_epoch(model, optimizer, loss_fn, X, y, batch_size, rng):
    model.train()
    idx = rng.permutation(X.shape[0])
    total_loss, n = 0.0, 0
    for i in range(0, X.shape[0], batch_size):
        b = idx[i:i + batch_size]
        xb = torch.from_numpy(X[b]).to(model.net[0].weight.device)
        yb = torch.from_numpy(y[b]).to(model.net[0].weight.device).float()
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
    return total_loss / max(n, 1)


def _eval_loss(model, loss_fn, X, y):
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).to(model.net[0].weight.device)
        yb = torch.from_numpy(y).to(model.net[0].weight.device).float()
        return float(loss_fn(model(xb), yb).item())


def run_experiment(
    episodes: List[EpisodeRecord],
    cfg: TrainConfig,
    *,
    remap_seed: bool,
    held_out_types: Optional[List[str]],
    tag: str,
) -> dict:
    set_seed(cfg.seed)
    eps = _unique_seed_episodes(episodes) if remap_seed else episodes
    split = split_episodes(
        eps, val_frac=cfg.val_frac, test_frac=cfg.test_frac,
        seed=cfg.split_seed, held_out_fault_types=held_out_types,
    )
    train_eps, val_eps, test_eps = split.train, split.val, split.test

    X_train, y_train, _ = episodes_to_arrays(train_eps)
    X_val, y_val, _ = episodes_to_arrays(val_eps)
    X_test, y_test, _ = episodes_to_arrays(test_eps)

    mean, std = compute_normalization_stats(X_train)  # 只从 train 计算 (防泄漏)
    X_train = normalize(X_train, mean, std)
    X_val = normalize(X_val, mean, std)
    X_test = normalize(X_test, mean, std)

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = float(np.clip(n_neg / max(n_pos, 1), 1.0, 50.0))

    device = cfg.device
    model = MLP(input_dim=cfg.input_dim, hidden_dims=cfg.hidden_dims, dropout=cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    rng = np.random.default_rng(cfg.seed)
    best_val_pr_auc = -1.0
    best_state = None
    hist: List[dict] = []
    for ep in range(cfg.epochs):
        train_loss = train_one_epoch(model, optimizer, loss_fn, X_train, y_train, cfg.batch_size, rng)
        val_loss = _eval_loss(model, loss_fn, X_val, y_val)
        score_val = predict_probs_batch(model, X_val, mean, std, device=device)
        val_pr_auc = average_precision(y_val, score_val)
        val_f1 = _f1_at(y_val, score_val, cfg.default_threshold)
        hist.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss,
                     "val_pr_auc": val_pr_auc, "val_f1": val_f1})
        if val_pr_auc is not None and val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 恢复最佳 checkpoint (按 val PR-AUC)。
    model.load_state_dict(best_state)

    # 阈值在 val 上按 F1 扫描。
    score_val = predict_probs_batch(model, X_val, mean, std, device=device)
    best_t, best_val_f1 = _select_threshold(y_val, score_val)

    # held-out test 评估 (模型)。
    predictor = MLPFailurePredictor(model, mean, std, threshold=best_t, device=device)
    test_model = _model_metrics(test_eps, predictor, threshold=best_t)
    test_model_05 = _model_metrics(test_eps, predictor, threshold=cfg.default_threshold)

    # 基线 (majority / rule)。
    baselines = _baseline_metrics(test_eps, train_eps)

    train_score = predict_probs_batch(model, X_train, mean, std, device=device)
    train_pr_auc = average_precision(y_train, train_score)

    return {
        "tag": tag,
        "split": split.to_dict(),
        "norm": {"mean": mean.tolist(), "std": std.tolist()},
        "pos_weight": pos_weight,
        "best_val_pr_auc": best_val_pr_auc,
        "best_val_f1": best_val_f1,
        "best_threshold": best_t,
        "train_pr_auc": train_pr_auc,
        "history": hist,
        "test": {
            "model@0.5": test_model_05,
            "model@tuned": test_model,
            "baselines": baselines,
        },
        "model": model,
        "mean": mean, "std": std,
    }


def _save_checkpoint(path_pt: str, path_json: str, res: dict, cfg: TrainConfig,
                     tag: str) -> None:
    model: MLP = res["model"]
    torch.save({
        "state_dict": model.state_dict(),
        "mean": res["mean"],
        "std": res["std"],
        "threshold": res["best_threshold"],
        "hidden_dims": model.hidden_dims,
        "input_dim": model.input_dim,
        "dropout": cfg.dropout,
        "config": cfg.to_dict(),
        "tag": tag,
        "val": {"pr_auc": res["best_val_pr_auc"], "f1": res["best_val_f1"]},
    }, path_pt)
    cfg_dict = cfg.to_dict()
    cfg_dict.update({
        "tag": tag,
        "mean": res["mean"].tolist(),
        "std": res["std"].tolist(),
        "threshold": res["best_threshold"],
        "pos_weight": res["pos_weight"],
        "best_val_pr_auc": res["best_val_pr_auc"],
        "best_val_f1": res["best_val_f1"],
        "split_summary": res["split"]["summary"],
    })
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False)


def _brief(res: dict) -> str:
    t = res["test"]["model@tuned"]["metrics"]
    b = res["test"]["baselines"]
    lines = [
        f"[{res['tag']}] test_eps={res['split']['summary']['test']['num_episodes']} "
        f"test_frames={res['split']['summary']['test']['num_frames']} "
        f"pos_frac={res['split']['summary']['test']['num_positive_frames']/max(res['split']['summary']['test']['num_frames'],1):.3f}",
        f"  best_val_pr_auc={res['best_val_pr_auc']} best_val_f1={res['best_val_f1']} "
        f"best_thr={res['best_threshold']:.3f} pos_weight={res['pos_weight']:.2f}",
        f"  model@tuned: P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f} "
        f"PR-AUC={t['pr_auc']} FP_rate={t['fp_rate']:.3f} FN_rate={t['fn_rate']:.3f} "
        f"latency={t['prediction_latency_mean_steps']} lead={t['lead_time_mean_steps']}",
        f"  majority : P={b['majority']['metrics']['precision']:.3f} R={b['majority']['metrics']['recall']:.3f} "
        f"F1={b['majority']['metrics']['f1']:.3f} PR-AUC={b['majority']['metrics']['pr_auc']}",
        f"  rule     : P={b['rule']['metrics']['precision']:.3f} R={b['rule']['metrics']['recall']:.3f} "
        f"F1={b['rule']['metrics']['f1']:.3f} PR-AUC={b['rule']['metrics']['pr_auc']} "
        f"FP_rate={b['rule']['metrics']['fp_rate']:.3f} FN_rate={b['rule']['metrics']['fn_rate']:.3f}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    set_seed(args.seed)

    episodes = load_episodes()
    print(f"loaded {len(episodes)} episodes")

    cfg = TrainConfig(seed=args.seed, epochs=args.epochs, device=args.device)

    # 主切分: held-out object_impulse (唯一 seed 重映射)。
    print("\n=== 主实验: held-out object_impulse (unique-seed remap) ===")
    res_held = run_experiment(
        episodes, cfg, remap_seed=True,
        held_out_types=list(cfg.held_out_fault_types), tag="heldout_object_impulse",
    )

    # 次切分: 真实 seed 的 disjoint-seed 泛化。
    print("\n=== 次实验: disjoint-seed (真实 seed) ===")
    res_seed = run_experiment(
        episodes, cfg, remap_seed=False, held_out_types=None, tag="seed_split",
    )

    _save_checkpoint(
        os.path.join(CHECKPOINT_DIR, "model.pt"),
        os.path.join(CHECKPOINT_DIR, "config.json"),
        res_held, cfg, "heldout_object_impulse",
    )
    _save_checkpoint(
        os.path.join(CHECKPOINT_DIR, "model_seed_split.pt"),
        os.path.join(CHECKPOINT_DIR, "config_seed_split.json"),
        res_seed, cfg, "seed_split",
    )

    # 训练曲线摘要 (首/中/末 epoch)。
    for res in (res_held, res_seed):
        h = res["history"]
        idxs = sorted({0, len(h) // 2, len(h) - 1})
        print(f"\n[{res['tag']}] 训练曲线 (epoch: train_loss/val_loss/val_pr_auc/val_f1):")
        for i in idxs:
            e = h[i]
            print(f"  ep {e['epoch']:3d}  train_loss={e['train_loss']:.4f} "
                  f"val_loss={e['val_loss']:.4f} val_pr_auc={e['val_pr_auc']} val_f1={e['val_f1']:.4f}")

    print("\n" + "=" * 80)
    print(_brief(res_held))
    print(_brief(res_seed))
    print("=" * 80)
    print(f"checkpoints -> {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
