"""M4-C 独立评估脚本: 从 checkpoint 重建预测器, 在 held-out test 上评估并对比基线。

用法 (仓库根目录):
  python3 learning/evaluate.py --checkpoint data/m4/checkpoints/model.pt \
      --config data/m4/checkpoints/config.json --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from typing import List, Optional

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from interfaces import EpisodeRecord  # noqa: E402
from eval.learning_split import split_episodes  # noqa: E402
from eval.learning_metrics import compute_prediction_metrics  # noqa: E402
from eval.learning_baselines import MajorityClassPredictor, RuleBasedPredictor  # noqa: E402
from learning.dataset import load_episodes  # noqa: E402
from learning.model import MLP, MLPFailurePredictor  # noqa: E402


def _unique_seed_episodes(episodes: List[EpisodeRecord]) -> List[EpisodeRecord]:
    return [replace(ep, seed=i) for i, ep in enumerate(episodes)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    ckpt_dir = os.path.join(_REPO, "data", "m4", "checkpoints")
    ckpt_path = args.checkpoint or os.path.join(ckpt_dir, "model.pt")
    cfg_path = args.config or os.path.join(ckpt_dir, "config.json")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)
    threshold = float(ckpt.get("threshold", 0.5))
    hidden_dims = list(ckpt.get("hidden_dims", cfg.get("hidden_dims", [32])))
    input_dim = int(ckpt.get("input_dim", cfg.get("input_dim", 22)))
    dropout = float(ckpt.get("dropout", cfg.get("dropout", 0.0)))

    model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    episodes = load_episodes()
    held_out = list(cfg.get("held_out_fault_types", ["object_impulse"]))
    val_frac = float(cfg.get("val_frac", 0.15))
    test_frac = float(cfg.get("test_frac", 0.15))
    split_seed = int(cfg.get("split_seed", 0))

    eps = _unique_seed_episodes(episodes)
    split = split_episodes(
        eps, val_frac=val_frac, test_frac=test_frac,
        seed=split_seed, held_out_fault_types=held_out,
    )

    predictor = MLPFailurePredictor(model, mean, std, threshold=threshold, device="cpu")
    test_eps = split.test
    train_eps = split.train

    model_tuned = compute_prediction_metrics(test_eps, predictor, threshold=threshold).to_dict()
    model_05 = compute_prediction_metrics(test_eps, predictor, threshold=0.5).to_dict()

    majority = MajorityClassPredictor.fit(train_eps)
    rule = RuleBasedPredictor()
    baselines = {
        "majority": compute_prediction_metrics(test_eps, majority, threshold=None).to_dict(),
        "rule": compute_prediction_metrics(test_eps, rule, threshold=None).to_dict(),
    }

    result = {
        "checkpoint": ckpt_path,
        "config": cfg_path,
        "threshold": threshold,
        "hidden_dims": hidden_dims,
        "split_summary": split.summary(),
        "test": {
            "model@tuned": model_tuned,
            "model@0.5": model_05,
            "baselines": baselines,
        },
    }

    out_json = json.dumps(result, indent=2, ensure_ascii=False)
    print(out_json)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(out_json)
        print(f"\n[written] {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
