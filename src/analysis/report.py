# src/analysis/full_report.py

import os
import argparse
from typing import Dict, List

import numpy as np
import torch

from src.configs import load_paths_config

TASK_NAMES = ["visit_mort", "patient_mort", "visit_readm", "patient_readm"]


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------

def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC, robust to constant / edge cases."""
    mask = ~np.isnan(y_true)
    y_true = y_true[mask].astype(int)
    y_score = y_score[mask]

    if y_true.size == 0 or np.all(y_true == 0) or np.all(y_true == 1):
        return float("nan")

    order = np.argsort(-y_score)
    y_true = y_true[order]

    pos = (y_true == 1).sum()
    neg = (y_true == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")

    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)
    tpr = tps / pos
    fpr = fps / neg

    return float(np.trapz(tpr, fpr))


def _binary_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUPRC (area under precision-recall)."""
    mask = ~np.isnan(y_true)
    y_true = y_true[mask].astype(int)
    y_score = y_score[mask]

    if y_true.size == 0 or np.all(y_true == 0):
        return float("nan")

    order = np.argsort(-y_score)
    y_true = y_true[order]

    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)
    total_pos = max(1, (y_true == 1).sum())

    recall = tp / total_pos
    precision = tp / np.maximum(1, tp + fp)

    return float(np.trapz(precision, recall))


def _accuracy_f1(y_true: np.ndarray, y_score: np.ndarray, thresh: float = 0.5) -> Dict[str, float]:
    mask = ~np.isnan(y_true)
    y_true = y_true[mask].astype(int)
    y_score = y_score[mask]

    if y_true.size == 0:
        return {"accuracy": float("nan"), "f1": float("nan")}

    y_pred = (y_score >= thresh).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))

    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)

    return {"accuracy": float(acc), "f1": float(f1)}


def _precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ks: List[int],
    as_fraction: bool = False,
) -> Dict[str, float]:
    """
    y_true: 0/1 labels for a single task (only valid entries).
    y_score: scores for same entries.
    ks: if as_fraction=False -> absolute K (e.g. 50, 100),
        if as_fraction=True  -> fraction of n (e.g. 0.05 for top 5%).
    """
    n = y_true.shape[0]
    if n == 0:
        return {str(k): float("nan") for k in ks}

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    out: Dict[str, float] = {}
    for k in ks:
        if as_fraction:
            kk = int(max(1, round(k * n)))
            key = f"{int(k*100)}pct"
        else:
            kk = int(min(max(1, k), n))
            key = f"{kk}"
        top = y_sorted[:kk]
        prec = float(top.mean()) if top.size > 0 else float("nan")
        out[key] = prec
    return out


# ---------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------

def compute_metrics_for_model(
    preds_path: str,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    test_ids: np.ndarray,
    topk_counts: List[int],
    topk_fracs: List[float],
) -> List[Dict[str, object]]:
    """
    Read one prediction file and compute metrics for each task.

    Expected prediction dict structure (from your existing pipeline):
      - probs: [N, T] probabilities
      - mask:  [N, T] bool/int mask where predictions are valid
      - model: model name (optional)
      - task_names: list of task names (optional)
    """
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Predictions not found: {preds_path}")

    preds = torch.load(preds_path, map_location="cpu")

    model_name = preds.get("model", os.path.basename(preds_path).replace(".pt", ""))
    probs = preds["probs"].numpy()                # [N, T]
    pred_mask = preds["mask"].numpy().astype(bool)

    labels_np = labels.numpy()
    label_mask_np = label_mask.numpy().astype(bool)

    num_tasks = labels_np.shape[1]
    task_names = preds.get("task_names", TASK_NAMES[:num_tasks])

    rows: List[Dict[str, object]] = []

    for t in range(num_tasks):
        tname = task_names[t] if t < len(task_names) else f"task_{t}"

        valid = label_mask_np[test_ids, t] & pred_mask[test_ids, t]
        idx = test_ids[valid]

        if idx.size == 0:
            rows.append(
                {
                    "model": model_name,
                    "task": tname,
                    "auroc": float("nan"),
                    "auprc": float("nan"),
                    "accuracy": float("nan"),
                    "f1": float("nan"),
                    "p_at_50": float("nan"),
                    "p_at_100": float("nan"),
                    "p_at_5pct": float("nan"),
                    "p_at_10pct": float("nan"),
                    "n": 0,
                    "n_pos": 0,
                }
            )
            continue

        y_true = labels_np[idx, t].astype(int)
        y_score = probs[idx, t]

        auroc = _binary_roc_auc(y_true, y_score)
        auprc = _binary_pr_auc(y_true, y_score)
        acc_f1 = _accuracy_f1(y_true, y_score)

        p_abs = _precision_at_k(y_true, y_score, topk_counts, as_fraction=False)
        p_frac = _precision_at_k(y_true, y_score, topk_fracs, as_fraction=True)

        rows.append(
            {
                "model": model_name,
                "task": tname,
                "auroc": auroc,
                "auprc": auprc,
                "accuracy": acc_f1["accuracy"],
                "f1": acc_f1["f1"],
                "p_at_50": p_abs.get("50", float("nan")),
                "p_at_100": p_abs.get("100", float("nan")),
                "p_at_5pct": p_frac.get("5pct", float("nan")),
                "p_at_10pct": p_frac.get("10pct", float("nan")),
                "n": int(idx.size),
                "n_pos": int((y_true == 1).sum()),
            }
        )

    return rows


def _load_test_ids(paths) -> np.ndarray:
    test_ids_path = os.path.join(paths.data_splits_dir, "test_ids.txt")
    if not os.path.exists(test_ids_path):
        raise FileNotFoundError(f"test_ids.txt not found at {test_ids_path}")
    with open(test_ids_path, "r") as f:
        test_ids = [int(line.strip()) for line in f if line.strip()]
    return np.array(test_ids, dtype=np.int64)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preds",
        nargs="+",
        required=True,
        help="Prediction .pt files (e.g., predictions/gcn_preds_test.pt predictions/graphsage_preds_test.pt)",
    )
    args = parser.parse_args()

    paths = load_paths_config()

    labels_path = os.path.join(paths.data_graph_dir, "labels_outcomes.pt")
    mask_path = os.path.join(paths.data_graph_dir, "labels_outcomes_mask.pt")
    if not (os.path.exists(labels_path) and os.path.exists(mask_path)):
        raise FileNotFoundError("labels_outcomes.pt or labels_outcomes_mask.pt missing in data/graph")

    labels = torch.load(labels_path, map_location="cpu")
    label_mask = torch.load(mask_path, map_location="cpu")
    test_ids = _load_test_ids(paths)
    print(f"Loaded {len(test_ids)} test_ids")

    topk_counts = [50, 100]
    topk_fracs = [0.05, 0.10]

    all_rows: List[Dict[str, object]] = []
    for p in args.preds:
        rows = compute_metrics_for_model(
            p,
            labels,
            label_mask,
            test_ids,
            topk_counts,
            topk_fracs,
        )
        all_rows.extend(rows)

    # Write TSV
    # os.makedirs(paths.results_dir, exist_ok=True)
    # out_path = os.path.join(paths.results_dir, "full_results.tsv")
    out_path = "src/results/full_results.tsv"

    header = [
        "model",
        "task",
        "auroc",
        "auprc",
        "accuracy",
        "f1",
        "p_at_50",
        "p_at_100",
        "p_at_5pct",
        "p_at_10pct",
        "n",
        "n_pos",
    ]

    with open(out_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in all_rows:
            def fmt(x):
                if isinstance(x, float) and np.isnan(x):
                    return "nan"
                if isinstance(x, float):
                    return f"{x:.6f}"
                return str(x)

            f.write("\t".join(fmt(r[h]) for h in header) + "\n")

    print(f"Wrote full results to {out_path}")


if __name__ == "__main__":
    main()
