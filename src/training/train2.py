# src/training/train_single.py

import os
import argparse

import numpy as np
import torch
import torch.nn as nn

import dgl
from dgl.dataloading import NeighborSampler, DataLoader

from src.configs import load_paths_config, load_training_config
from src.models.graphsage import GraphSAGE
from src.models.gcn import GCN


def _get_device() -> torch.device:
    # CPU on your Mac right now; will automatically use GPU later if available
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def _build_model(
    model_name: str,
    num_nodes: int,
    out_dim: int,
    model_cfg,
    device: torch.device,
) -> nn.Module:
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
        raise ValueError(f"Unknown model_name: {model_name} (expected 'graphsage' or 'gcn')")

    return model.to(device)


def _compute_pos_weight(labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute per-task pos_weight for BCEWithLogitsLoss:
      pos_weight_t = N_neg / N_pos  over masked entries.
    """
    labels_np = labels.numpy()
    mask_np = mask.numpy().astype(bool)

    num_tasks = labels_np.shape[1]
    weights = []

    for t in range(num_tasks):
        m = mask_np[:, t]
        y = labels_np[m, t]
        if y.size == 0:
            w = 1.0
        else:
            pos = int((y == 1).sum())
            neg = int((y == 0).sum())
            w = float(neg / max(1, pos)) if pos > 0 else 1.0
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[None, "graphsage", "gcn"],
        help="Optional override; if None, use training.yaml:model.type",
    )
    args = parser.parse_args()

    device = _get_device()
    print(f"[Single-process] Using device: {device}")

    paths = load_paths_config()
    train_cfg = load_training_config()

    # Decide model type: CLI override or YAML
    model_name = args.model if args.model is not None else train_cfg.model.type
    print(f"Model type: {model_name}")

    # ---------------------------
    # 1. Load graph
    # ---------------------------
    graph_path = os.path.join(paths.data_graph_dir, "dgl", "full_graph.bin")
    print(f"Loading graph from {graph_path}")
    glist, _ = dgl.load_graphs(graph_path)
    g = glist[0]

    num_nodes = g.num_nodes()
    num_edges = g.num_edges()
    print(f"Graph: {num_nodes} nodes, {num_edges} edges")

    # ---------------------------
    # 2. Load labels/mask + train_ids
    # ---------------------------
    labels_path = os.path.join(paths.data_graph_dir, "labels_outcomes.pt")
    mask_path = os.path.join(paths.data_graph_dir, "labels_outcomes_mask.pt")
    train_ids_path = os.path.join(paths.data_splits_dir, "train_ids.txt")

    if not (os.path.exists(labels_path) and os.path.exists(mask_path) and os.path.exists(train_ids_path)):
        print("ERROR: labels_outcomes / labels_outcomes_mask / train_ids not found.")
        print("Run src.preprocessing.build_outcome_labels_splits first.")
        return

    labels = torch.load(labels_path, map_location="cpu")        # [num_nodes, num_tasks]
    label_mask = torch.load(mask_path, map_location="cpu")      # [num_nodes, num_tasks]

    with open(train_ids_path, "r") as f:
        train_ids = [int(line.strip()) for line in f if line.strip()]
    train_ids = torch.tensor(train_ids, dtype=torch.long)

    num_labels = labels.shape[1]
    print(f"Num tasks (label dims): {num_labels}")
    print(f"Total train_ids: {len(train_ids)}")

    # ---------------------------
    # 3. Compute pos_weight and inspect class balance
    # ---------------------------
    pos_weight = _compute_pos_weight(labels, label_mask)
    print("pos_weight per task:", pos_weight.tolist())

    # Optional: quick class-balance dump
    labels_np = labels.numpy()
    mask_np = label_mask.numpy().astype(bool)
    for t in range(num_labels):
        m = mask_np[:, t]
        y = labels_np[m, t]
        if y.size == 0:
            print(f"Task {t}: no labeled entries")
            continue
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        print(f"Task {t}: pos={pos}, neg={neg}, pos_rate={pos/(pos+neg):.4f}")

    # ---------------------------
    # 4. Model + optimizer
    # ---------------------------
    model = _build_model(
        model_name=model_name,
        num_nodes=num_nodes,
        out_dim=num_labels,
        model_cfg=train_cfg.model,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.training.lr,
        weight_decay=train_cfg.training.weight_decay,
    )

    # ---------------------------
    # 5. NeighborSampler + DataLoader
    # ---------------------------
    fanouts = list(train_cfg.sampling.fanouts)   # e.g., [10, 10]
    print(f"NeighborSampler fanouts: {fanouts}")

    sampler = NeighborSampler(fanouts)

    dataloader = DataLoader(
        g,
        train_ids,
        sampler,
        batch_size=train_cfg.training.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=train_cfg.sampling.num_workers,
        device=device,   # DGL will move blocks/tensors to this device
    )

    # ---------------------------
    # 6. Training loop with masked, pos-weighted BCE
    # ---------------------------
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction="none")
    num_epochs = train_cfg.training.num_epochs

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for input_nodes, output_nodes, blocks in dataloader:
            # Already on `device` because of device=device in DataLoader
            logits = model(blocks, input_nodes)          # [B, num_labels]
            y = labels[output_nodes].to(device)          # [B, num_labels]
            m = label_mask[output_nodes].to(device)      # [B, num_labels]

            raw_loss = loss_fn(logits, y)                # [B, num_labels]
            weighted = raw_loss * m                      # mask unlabeled tasks

            denom = m.sum().clamp(min=1.0)
            loss = weighted.sum() / denom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        print(f"[Epoch {epoch}] model={model_name} avg_loss={avg_loss:.4f}")

    print("Training finished.")

    # ---------------------------
    # 7. Save checkpoint
    # ---------------------------
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(project_root, "models")
    os.makedirs(ckpt_dir, exist_ok=True)

    out_ckpt = os.path.join(ckpt_dir, f"{model_name}_mort_readm_allparts.pt")
    torch.save(model.state_dict(), out_ckpt)
    print(f"Saved model checkpoint to: {out_ckpt}")


if __name__ == "__main__":
    main()
