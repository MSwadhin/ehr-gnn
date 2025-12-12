# src/analysis/full_report.py

import os
import argparse
from typing import Dict, List

import numpy as np
import torch

from src.configs import load_paths_config

# Default task names (fallback)
TASK_NAMES = ["visit_mort", "patient_mort", "visit_readm", "patient_readm"]


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------

def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC with basic edge-case handling."""
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
    """AUPRC: area under precision-recall curve."""
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
    """Accuracy and F1 at a fixed probability threshold."""
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

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)

    return {"accuracy": float(acc), "f1": float(f1)}


def _precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ks,
    frac: bool = False,
) -> Dict[str, float]:
    """
    Precision@K.

    If frac == False: ks are absolute integers (e.g., 50, 100).
    If frac == True:  ks are fractions of n (e.g., 0.05 for top 5%).
    """
    n = y_true.shape[0]
    if n == 0:
        return {str(k): float("nan") for k in ks}

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    out: Dict[str, float] = {}

    for k in ks:
        if frac:
            k2 = int(max(1, round(k * n)))
            key = f"{int(k * 100)}pct"
        else:
            k2 = min(max(1, int(k)), n)
            key = str(k)

        top = y_sorted[:k2]
        prec = float(top.mean()) if top.size > 0 else float("nan")
        out[key] = prec

    return out


# ---------------------------------------------------------------------
# Core computation for a single prediction file
# ---------------------------------------------------------------------

def compute_metrics_for_model(
    preds_path: str,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    test_ids: np.ndarray,
) -> List[Dict[str, object]]:
    """
    Compute metrics (AUROC, AUPRC, accuracy, F1, precision@K, n, n_pos)
    for each task from one prediction file.

    Expected pred file structure:
      {
        "model": str (optional),
        "split": str (optional),
        "probs": FloatTensor [N, T],
        "mask":  BoolTensor [N, T],
        "task_names": list[str] (optional)
      }
    """
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Predictions not found: {preds_path}")

    preds = torch.load(preds_path, map_location="cpu")

    model_name = preds.get("model", os.path.basename(preds_path).replace(".pt", ""))
    probs = preds["probs"].numpy()                # [N, T]
    pred_mask = preds["mask"].numpy().astype(bool)

    labels_np = labels.numpy()
    mask_np = label_mask.numpy().astype(bool)

    num_tasks = labels_np.shape[1]
    task_names = preds.get("task_names", TASK_NAMES[:num_tasks])

    rows: List[Dict[str, object]] = []

    for t in range(num_tasks):
        task = task_names[t] if t < len(task_names) else f"task_{t}"

        # Valid when both label and prediction available
        valid = mask_np[test_ids, t] & pred_mask[test_ids, t]
        idx = test_ids[valid]

        if idx.size == 0:
            rows.append(
                {
                    "model": model_name,
                    "task": task,
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
        accf1 = _accuracy_f1(y_true, y_score)

        p_abs = _precision_at_k(y_true, y_score, [50, 100], frac=False)
        p_frac = _precision_at_k(y_true, y_score, [0.05, 0.10], frac=True)

        rows.append(
            {
                "model": model_name,
                "task": task,
                "auroc": auroc,
                "auprc": auprc,
                "accuracy": accf1["accuracy"],
                "f1": accf1["f1"],
                "p_at_50": p_abs["50"],
                "p_at_100": p_abs["100"],
                "p_at_5pct": p_frac["5pct"],
                "p_at_10pct": p_frac["10pct"],
                "n": int(idx.size),
                "n_pos": int((y_true == 1).sum()),
            }
        )

    return rows


def _load_test_ids(paths) -> np.ndarray:
    """Load integer node IDs for test split from data/splits/test_ids.txt."""
    test_ids_path = os.path.join(paths.data_splits_dir, "test_ids.txt")
    if not os.path.exists(test_ids_path):
        raise FileNotFoundError(f"test_ids.txt not found at {test_ids_path}")
    with open(test_ids_path, "r") as f:
        test_ids = [int(x.strip()) for x in f if x.strip()]
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
        help="Prediction .pt files, e.g.: predictions/gcn_preds_test.pt predictions/graphsage_preds_test.pt",
    )
    args = parser.parse_args()

    paths = load_paths_config()

    # Load labels + mask
    labels_path = os.path.join(paths.data_graph_dir, "labels_outcomes.pt")
    mask_path = os.path.join(paths.data_graph_dir, "labels_outcomes_mask.pt")
    if not (os.path.exists(labels_path) and os.path.exists(mask_path)):
        raise FileNotFoundError("labels_outcomes.pt or labels_outcomes_mask.pt missing in data/graph")

    labels = torch.load(labels_path, map_location="cpu")
    label_mask = torch.load(mask_path, map_location="cpu")

    # Load test IDs
    test_ids = _load_test_ids(paths)
    print(f"Loaded {len(test_ids)} test_ids")

    # Aggregate results from all prediction files
    all_rows: List[Dict[str, object]] = []
    for p in args.preds:
        rows = compute_metrics_for_model(p, labels, label_mask, test_ids)
        all_rows.extend(rows)

    # Hardcoded results dir in project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    out_tsv = os.path.join(results_dir, "full_results.tsv")

    header = [
        "model", "task",
        "auroc", "auprc",
        "accuracy", "f1",
        "p_at_50", "p_at_100",
        "p_at_5pct", "p_at_10pct",
        "n", "n_pos",
    ]

    # Write TSV
    with open(out_tsv, "w") as f:
        f.write("\t".join(header) + "\n")

        for r in all_rows:
            def fmt(x):
                if isinstance(x, float) and np.isnan(x):
                    return "nan"
                if isinstance(x, float):
                    return f"{x:.6f}"
                return str(x)

            f.write("\t".join(fmt(r[h]) for h in header) + "\n")

    print(f"Wrote full results to {out_tsv}")


if __name__ == "__main__":
    main()
