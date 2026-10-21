"""B2 vs B2-T (phase 门控) 消融对比: 加载 B2 与 B2-T 的 episode 记录, matched 配对对比 + paired bootstrap。

用法:
  python scripts/compare_b2t.py --b2 logs/m5_primary_episodes.jsonl --b2t logs/m5_b2t_episodes.jsonl --out logs/m5_b2t_comparison.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def load(jsonl, method):
    recs = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("method") == method:
                recs.append(r)
    return recs


def wilson(k, n):
    if n == 0:
        return None, None
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def paired_bootstrap(diffs, n_boot=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(diffs), len(diffs))
        means.append(diffs[idx].mean())
    means = np.array(means)
    return {
        "mean": float(diffs.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "n": len(diffs),
        "distribution": means.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b2", required=True)
    ap.add_argument("--b2t", required=True)
    ap.add_argument("--out", default="logs/m5_b2t_comparison.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    b2 = {(r["fault"], int(r["seed"])): r for r in load(args.b2, "b2")}
    b2t = {(r["fault"], int(r["seed"])): r for r in load(args.b2t, "b2t")}
    common = sorted(set(b2) & set(b2t))

    diffs, b2_succ, b2t_succ = [], 0, 0
    b2_rec = b2t_rec = b2_den = b2t_den = 0
    b2_trig = b2t_trig = 0
    b2_abort = b2t_abort = 0
    b2_retries = b2t_retries = []
    false_trigger_eps = 0  # b2t 触发 但 b2 同 seed 首次尝试即成功(无需恢复)
    rows = []
    for key in common:
        a, b = b2[key], b2t[key]
        diffs.append(int(b["success"]) - int(a["success"]))
        b2_succ += int(a["success"]); b2t_succ += int(b["success"])
        b2_trig += a.get("num_triggers", 0); b2t_trig += b.get("num_triggers", 0)
        b2_abort += int(a.get("recovery_aborted", False))
        b2t_abort += int(b.get("recovery_aborted", False))
        if a.get("retries") is not None:
            b2_retries.append(a["retries"])
        if b.get("retries") is not None:
            b2t_retries.append(b["retries"])
        # false trigger: b2t 触发但 b2(单帧) 首次尝试即成功
        if b.get("num_triggers", 0) > 0 and a.get("attempts", 1) == 1 and a.get("success"):
            false_trigger_eps += 1
        # recovery denominator: injected+manifested
        if a.get("verified") and a.get("manifested"):
            b2_den += 1; b2_rec += int(a.get("recovery_recovered", False))
        if b.get("verified") and b.get("manifested"):
            b2t_den += 1; b2t_rec += int(b.get("recovery_recovered", False))
        rows.append({
            "fault": key[0], "seed": key[1],
            "b2_success": int(a["success"]), "b2t_success": int(b["success"]),
            "b2_triggers": a.get("num_triggers", 0), "b2t_triggers": b.get("num_triggers", 0),
            "b2_attempts": a.get("attempts"), "b2t_attempts": b.get("attempts"),
            "b2_aborted": int(a.get("recovery_aborted", False)),
            "b2t_aborted": int(b.get("recovery_aborted", False)),
        })

    n = len(common)
    pb = paired_bootstrap(diffs, args.n_boot)
    b2_lo, b2_hi = wilson(b2_succ, n)
    b2t_lo, b2t_hi = wilson(b2t_succ, n)

    report = {
        "n_matched_units": n,
        "b2": {"task_success": b2_succ, "rate": b2_succ / n if n else None,
               "wilson_ci": [b2_lo, b2_hi], "total_triggers": b2_trig,
               "aborted": b2_abort, "avg_retries": float(np.mean(b2_retries)) if b2_retries else None,
               "recovery_success": f"{b2_rec}/{b2_den}" if b2_den else None},
        "b2t": {"task_success": b2t_succ, "rate": b2t_succ / n if n else None,
                "wilson_ci": [b2t_lo, b2t_hi], "total_triggers": b2t_trig,
                "aborted": b2t_abort, "avg_retries": float(np.mean(b2t_retries)) if b2t_retries else None,
                "recovery_success": f"{b2t_rec}/{b2t_den}" if b2t_den else None},
        "paired_diff_b2t_minus_b2": pb,
        "false_trigger_episodes_b2t": false_trigger_eps,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"B2   vs B2-T  (matched {n} units)")
    print(f"  task_success:  B2 {b2_succ}/{n}={b2_succ/n:.3f}  B2-T {b2t_succ}/{n}={b2t_succ/n:.3f}")
    print(f"  paired diff B2T-B2: mean={pb['mean']:.3f}  95%CI [{pb['ci_lo']:.3f}, {pb['ci_hi']:.3f}]")
    print(f"  total_triggers: B2 {b2_trig}  B2-T {b2t_trig}")
    print(f"  aborted: B2 {b2_abort}  B2-T {b2t_abort}")
    print(f"  recovery_success: B2 {report['b2']['recovery_success']}  B2-T {report['b2t']['recovery_success']}")
    print(f"  false_trigger_eps(b2t): {false_trigger_eps}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
