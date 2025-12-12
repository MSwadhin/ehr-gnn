import os
import torch
import numpy as np

from src.configs import load_paths_config

TASK_NAMES = ["visit_mort", "patient_mort", "visit_readm", "patient_readm"]


def main():
    paths = load_paths_config()

    # ---- Load labels + mask (same as full_report) ----
    labels_path = os.path.join(paths.data_graph_dir, "labels_outcomes.pt")
    mask_path = os.path.join(paths.data_graph_dir, "labels_outcomes_mask.pt")

    labels = torch.load(labels_path, map_location="cpu")
    label_mask = torch.load(mask_path, map_location="cpu")

    labels_np = labels.numpy()
    label_mask_np = label_mask.numpy().astype(bool)

    # ---- Load both prediction files ----
    gcn = torch.load("src/predictions/gcn_preds_test.pt", map_location="cpu")
    sage = torch.load("src/predictions/graphsage_preds_test.pt", map_location="cpu")

    probs_gcn = gcn["probs"].numpy()
    probs_sage = sage["probs"].numpy()
    mask_gcn = gcn["mask"].numpy().astype(bool)
    mask_sage = sage["mask"].numpy().astype(bool)

    print("GCN model field:", gcn.get("model"))
    print("GraphSAGE model field:", sage.get("model"))

    print("Global allclose(probs):", np.allclose(probs_gcn, probs_sage))

    num_tasks = labels_np.shape[1]
    for t in range(num_tasks):
        tname = TASK_NAMES[t] if t < len(TASK_NAMES) else f"task_{t}"

        valid_gcn = label_mask_np[:, t] & mask_gcn[:, t]
        valid_sage = label_mask_np[:, t] & mask_sage[:, t]
        valid_both = valid_gcn & valid_sage

        if valid_both.sum() == 0:
            print(f"\nTask {t} ({tname}): no valid nodes for both models")
            continue

        p_gcn = probs_gcn[valid_both, t]
        p_sage = probs_sage[valid_both, t]

        diff = np.abs(p_gcn - p_sage)
        print(f"\nTask {t} ({tname})")
        print("  valid_both:", int(valid_both.sum()))
        print("  mean |GCN - SAGE|:", float(diff.mean()))
        print("  max  |GCN - SAGE|:", float(diff.max()))
        print("  corr(GCN, SAGE):", float(np.corrcoef(p_gcn, p_sage)[0, 1]))

        # simple sanity: compute AUROC difference manually using current metrics code
        from src.analysis.full_report import _binary_roc_auc

        y_true = labels_np[valid_both, t].astype(int)
        auroc_gcn = _binary_roc_auc(y_true, p_gcn)
        auroc_sage = _binary_roc_auc(y_true, p_sage)
        print("  AUROC GCN vs SAGE:", auroc_gcn, auroc_sage)


if __name__ == "__main__":
    main()
