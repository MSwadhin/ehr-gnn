# src/evaluation/predict.py

import os
import argparse

import torch
import torch.nn as nn

import dgl
from dgl.dataloading import NeighborSampler, DataLoader

from src.configs import load_paths_config, load_training_config
from src.models.graphsage import GraphSAGE
from src.models.gcn import GCN
from src.utils.io_utils import read_parquet  # optional, for partition info


TASK_NAMES = ["visit_mort", "patient_mort", "visit_readm", "patient_readm"]


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def _build_model(model_name, num_nodes, out_dim, model_cfg, device):
    model_name = model_name.lower()
    if model_name == "graphsage":
        model = GraphSAGE(
            num_nodes=num_nodes,
            hidden_dim=model_cfg.hidden_dim,
            out_dim=out_dim,
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
            aggregator_type=getattr(model_cfg, "aggregator", "mean"),
        )
    elif model_name == "gcn":
        model = GCN(
            num_nodes=num_nodes,
            hidden_dim=model_cfg.hidden_dim,
            out_dim=out_dim,
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
            norm="both",
        )
    else:
        raise ValueError(f"Unknown model_name {model_name} (expected graphsage|gcn)")
    return model.to(device)


def _load_split_ids(paths, split: str):
    if split == "all":
        return None  # we’ll handle this later
    fname = f"{split}_ids.txt"
    split_path = os.path.join(paths.data_splits_dir, fname)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with open(split_path, "r") as f:
        ids = [int(line.strip()) for line in f if line.strip()]
    return torch.tensor(ids, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gcn",
        choices=["graphsage", "gcn"],
        help="Model architecture (must match training).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to checkpoint (.pt). If not set, use models/<model>_mort_readm_allparts.pt",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which node split to run predictions on.",
    )
    args = parser.parse_args()

    device = _get_device()
    print(f"[Predict] Using device: {device}")
    print(f"Model: {args.model}, split: {args.split}")

    paths = load_paths_config()
    train_cfg = load_training_config()

    # ---------------------------
    # 1. Load graph (+ optionally partition info)
    # ---------------------------
    graph_path = os.path.join(paths.data_graph_dir, "dgl", "full_graph.bin")
    print(f"Loading graph from {graph_path}")
    glist, _ = dgl.load_graphs(graph_path)
    g = glist[0]

    num_nodes = g.num_nodes()
    print(f"Graph: {num_nodes} nodes, {g.num_edges()} edges")

    # Optional: partitions / node types, for output/meta
    nodes_meta_path = os.path.join(paths.data_graph_dir, "dgl", "nodes_meta.parquet")
    nodes_meta = None
    if os.path.exists(nodes_meta_path):
        nodes_meta = read_parquet(nodes_meta_path)
        print("Loaded nodes_meta with columns:", list(nodes_meta.columns))

    # ---------------------------
    # 2. Load labels + mask (for potential evaluation)
    # ---------------------------
    labels_path = os.path.join(paths.data_graph_dir, "labels_outcomes.pt")
    mask_path = os.path.join(paths.data_graph_dir, "labels_outcomes_mask.pt")

    if not (os.path.exists(labels_path) and os.path.exists(mask_path)):
        print("WARNING: labels_outcomes or labels_outcomes_mask missing. "
              "Will still run predictions, but cannot compute losses/metrics.")
        labels = None
        label_mask = None
    else:
        labels = torch.load(labels_path, map_location="cpu")
        label_mask = torch.load(mask_path, map_location="cpu")

    num_labels = labels.shape[1] if labels is not None else len(TASK_NAMES)
    print(f"Num tasks (label dims): {num_labels}")

    # ---------------------------
    # 3. Decide which nodes to predict on
    # ---------------------------
    if args.split == "all":
        node_ids = torch.arange(num_nodes, dtype=torch.long)
    else:
        node_ids = _load_split_ids(paths, args.split)
        print(f"{args.split} nodes: {len(node_ids)}")

    # ---------------------------
    # 4. Build model + load checkpoint
    # ---------------------------
    model = _build_model(
        model_name=args.model,
        num_nodes=num_nodes,
        out_dim=num_labels,
        model_cfg=train_cfg.model,
        device=device,
    )

    if args.ckpt is None:
        # default checkpoint path: <project_root>/models/<model>_mort_readm_allparts.pt
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_dir = os.path.join(project_root, "models")
        ckpt_path = os.path.join(ckpt_dir, f"{args.model}_mort_readm_allparts.pt")
    else:
        ckpt_path = args.ckpt

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ---------------------------
    # 5. NeighborSampler + DataLoader for inference
    # ---------------------------
    fanouts = list(train_cfg.sampling.fanouts)
    print(f"NeighborSampler fanouts (inference): {fanouts}")
    sampler = NeighborSampler(fanouts)

    dataloader = DataLoader(
        g,
        node_ids,
        sampler,
        batch_size=train_cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=train_cfg.sampling.num_workers,
        device=device,
    )

    # ---------------------------
    # 6. Run predictions
    # ---------------------------
    all_logits = torch.zeros((num_nodes, num_labels), dtype=torch.float32)
    all_probs = torch.zeros((num_nodes, num_labels), dtype=torch.float32)
    all_mask = torch.zeros((num_nodes, num_labels), dtype=torch.bool)

    if labels is not None and label_mask is not None:
        labels = labels.to(device)
        label_mask = label_mask.to(device)

    loss_fn = nn.BCEWithLogitsLoss(reduction="none") if labels is not None else None
    total_loss = 0.0
    total_count = 0.0

    with torch.no_grad():
        for input_nodes, output_nodes, blocks in dataloader:
            logits = model(blocks, input_nodes)              # [B, num_labels]
            probs = torch.sigmoid(logits)

            # Move to CPU for storage
            output_nodes_cpu = output_nodes.cpu()
            logits_cpu = logits.cpu()
            probs_cpu = probs.cpu()

            all_logits[output_nodes_cpu] = logits_cpu
            all_probs[output_nodes_cpu] = probs_cpu
            all_mask[output_nodes_cpu] = True

            # Optional loss on labeled nodes in this split
            if labels is not None and label_mask is not None:
                y = labels[output_nodes].to(device)
                m = label_mask[output_nodes].to(device)

                raw_loss = loss_fn(logits, y)
                weighted = raw_loss * m
                denom = m.sum().clamp(min=1.0)
                batch_loss = weighted.sum() / denom

                total_loss += batch_loss.item()
                total_count += 1.0

    if total_count > 0:
        avg_loss = total_loss / total_count
        print(f"[{args.split}] avg BCE loss (on labeled tasks only): {avg_loss:.4f}")
    else:
        print(f"[{args.split}] No labeled nodes encountered; skipped loss computation.")

    # ---------------------------
    # 7. Save predictions
    # ---------------------------
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pred_dir = os.path.join(project_root, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    out_path = os.path.join(
        pred_dir,
        f"{args.model}_preds_{args.split}.pt",
    )

    out = {
        "model": args.model,
        "split": args.split,
        "task_names": TASK_NAMES[:num_labels],
        "logits": all_logits,    # [num_nodes, num_labels]
        "probs": all_probs,      # [num_nodes, num_labels]
        "mask": all_mask,        # which nodes we actually scored
    }

    if nodes_meta is not None:
        # Attach partition and node_type arrays for convenience (on CPU)
        out["node_type"] = nodes_meta["node_type"].to_numpy()
        out["partition"] = nodes_meta["partition"].to_numpy()

    torch.save(out, out_path)
    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
