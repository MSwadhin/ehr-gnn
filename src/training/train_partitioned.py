import os
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import dgl
from dgl.dataloading import NeighborSampler, DataLoader

from src.configs import load_paths_config, load_training_config
from src.utils.io_utils import read_parquet
from src.models.graphsage import GraphSAGE
from src.models.gcn import GCN


def _get_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    else:
        return torch.device("cpu")


def _get_partition_assignment(num_parts: int, world_size: int, rank: int) -> List[int]:
    """Simple partition -> worker mapping: round-robin."""
    parts = list(range(num_parts))
    return [p for i, p in enumerate(parts) if i % world_size == rank]


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


def train_worker(rank: int, world_size: int, args):
    # ---------------------------
    # 1. Distributed init
    # ---------------------------
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    device = _get_device(rank)
    if rank == 0:
        print(f"[Rank {rank}] Using device: {device}")

    paths = load_paths_config()
    train_cfg = load_training_config()

    # decide model type: CLI override or YAML
    model_name = args.model if args.model is not None else train_cfg.model.type

    # ---------------------------
    # 2. Load DGL graph + node meta
    # ---------------------------
    graph_path = os.path.join(paths.data_graph_dir, "dgl", "full_graph.bin")
    nodes_meta_path = os.path.join(paths.data_graph_dir, "dgl", "nodes_meta.parquet")

    if rank == 0:
        print(f"[Rank {rank}] Loading graph from {graph_path}")
    glist, _ = dgl.load_graphs(graph_path)
    g = glist[0]

    nodes_meta = read_parquet(nodes_meta_path)
    num_nodes = g.num_nodes()

    # ---------------------------
    # 3. Load outcome labels + mask + train_ids
    # ---------------------------
    labels_path = os.path.join(paths.data_graph_dir, "labels_outcomes.pt")
    mask_path = os.path.join(paths.data_graph_dir, "labels_outcomes_mask.pt")
    train_ids_path = os.path.join(paths.data_splits_dir, "train_ids.txt")

    if not (os.path.exists(labels_path) and os.path.exists(mask_path) and os.path.exists(train_ids_path)):
        if rank == 0:
            print(
                "ERROR: outcome labels/mask or train_ids not found. "
                "Run build_outcome_labels_splits.py first."
            )
        dist.destroy_process_group()
        return

    labels = torch.load(labels_path, map_location="cpu")        # [num_nodes, num_tasks]
    label_mask = torch.load(mask_path, map_location="cpu")      # [num_nodes, num_tasks]

    with open(train_ids_path, "r") as f:
        train_ids = [int(line.strip()) for line in f if line.strip()]
    train_ids = torch.tensor(train_ids, dtype=torch.long)

    num_labels = labels.shape[1]

    # ---------------------------
    # 4. Partition-aware node selection
    # ---------------------------
    part_col = nodes_meta["partition"]
    num_parts = int(part_col.max() + 1)
    my_parts = _get_partition_assignment(num_parts, world_size, rank)

    if rank == 0:
        print(f"World size={world_size}, num_parts={num_parts}")
        print(f"Rank {rank} will train on partitions: {my_parts}")

    part_tensor = torch.from_numpy(part_col.to_numpy())
    mask_part = torch.isin(part_tensor, torch.tensor(my_parts, dtype=torch.long))
    local_train_ids = train_ids[mask_part[train_ids]]

    if len(local_train_ids) == 0:
        if rank == 0:
            print(f"[Rank {rank}] No training nodes assigned to these partitions.")
        dist.destroy_process_group()
        return

    if rank == 0:
        print(f"[Rank {rank}] Local training nodes: {len(local_train_ids)}")

    # ---------------------------
    # 5. Model, optimizer, DDP
    # ---------------------------
    model = _build_model(
        model_name=model_name,
        num_nodes=num_nodes,
        out_dim=num_labels,
        model_cfg=train_cfg.model,
        device=device,
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=None if device.type == "cpu" else [device],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.training.lr,
        weight_decay=train_cfg.training.weight_decay,
    )

    # ---------------------------
    # 6. DGL NeighborSampler + DataLoader
    # ---------------------------
    fanouts = list(train_cfg.sampling.fanouts)
    sampler = NeighborSampler(fanouts)

    dataloader = DataLoader(
        g,
        local_train_ids,
        sampler,
        batch_size=train_cfg.training.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=train_cfg.sampling.num_workers,
        device=device,   # DGL moves blocks/tensors to device
    )

    # ---------------------------
    # 7. Training loop with masked BCE
    # ---------------------------
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    num_epochs = train_cfg.training.num_epochs

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for input_nodes, output_nodes, blocks in dataloader:
            # already on device because of `device=device` in DataLoader
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

        # Average loss across workers
        loss_tensor = torch.tensor(total_loss / max(1, total_batches), device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            avg_loss = loss_tensor.item() / world_size
            print(f"[Epoch {epoch}] model={model_name} avg_loss={avg_loss:.4f}")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[None, "graphsage", "gcn"],
        help="Optional override of model type; if None, use training.yaml:model.type",
    )
    args = parser.parse_args()

    world_size = args.world_size
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
