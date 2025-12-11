import os
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import dgl
from dgl.dataloading import NeighborSampler, NodeDataLoader

from src.configs import load_paths_config, load_training_config
from src.utils.io_utils import read_parquet
from src.training.models.graphsage import GraphSAGE


def _get_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    else:
        return torch.device("cpu")


def _get_partition_assignment(num_parts: int, world_size: int, rank: int) -> List[int]:
    """
    Simple partition -> worker mapping: split partitions as evenly as possible.
    """
    parts = list(range(num_parts))
    return [p for i, p in enumerate(parts) if i % world_size == rank]


def train_worker(rank: int, world_size: int, args):
    # ---------------------------
    # 1. Distributed init
    # ---------------------------
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    device = _get_device(rank)
    if rank == 0:
        print(f"[Rank {rank}] Using device: {device}")

    paths = load_paths_config()
    train_cfg = load_training_config()

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
    # 3. Load labels + splits (placeholder)
    # ---------------------------
    # TODO: replace this with your real label loading logic.
    # For now, we assume:
    #   - labels.pt: tensor [num_nodes, num_labels] (multi-hot)
    #   - train_ids.txt: node IDs (one per line)
    labels_path = os.path.join(paths.data_graph_dir, "labels.pt")
    train_ids_path = os.path.join(paths.data_splits_dir, "train_ids.txt")

    if not os.path.exists(labels_path) or not os.path.exists(train_ids_path):
        if rank == 0:
            print(
                "WARNING: labels.pt or train_ids.txt not found. "
                "Please implement your label+split loading and re-run."
            )
        dist.destroy_process_group()
        return

    labels = torch.load(labels_path, map_location="cpu")  # [num_nodes, num_labels]
    with open(train_ids_path, "r") as f:
        train_ids = [int(line.strip()) for line in f if line.strip()]

    train_ids = torch.tensor(train_ids, dtype=torch.long)

    # ---------------------------
    # 4. Partition-aware node selection
    # ---------------------------
    # Determine how many partitions exist and assign some to this rank.
    part_col = nodes_meta["partition"]
    num_parts = int(part_col.max() + 1)
    my_parts = _get_partition_assignment(num_parts, world_size, rank)

    if rank == 0:
        print(f"World size={world_size}, num_parts={num_parts}")
        print(f"Rank {rank} will train on partitions: {my_parts}")

    part_tensor = torch.from_numpy(part_col.to_numpy())
    mask = torch.isin(part_tensor, torch.tensor(my_parts, dtype=torch.long))
    local_train_ids = train_ids[mask[train_ids]]

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
    num_labels = labels.shape[1]
    model = GraphSAGE(
        num_nodes=num_nodes,
        hidden_dim=train_cfg.model.hidden_dim,
        out_dim=num_labels,
        num_layers=train_cfg.model.num_layers,
        dropout=train_cfg.model.dropout,
    ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=None if device.type == "cpu" else [device]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.optimizer.lr,
        weight_decay=train_cfg.optimizer.weight_decay,
    )

    # ---------------------------
    # 6. DGL NeighborSampler + DataLoader
    # ---------------------------
    fanouts = train_cfg.sampling.fanouts  # e.g., [10, 10]
    sampler = NeighborSampler(fanouts)

    # No distributed sampler; partitions already separate the nodes.
    dataloader = NodeDataLoader(
        g,
        local_train_ids,
        sampler,
        batch_size=train_cfg.training.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        device=device,
    )

    # ---------------------------
    # 7. Training loop
    # ---------------------------
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(train_cfg.training.num_epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device) for b in blocks]
            input_nodes = input_nodes.to(device)
            output_nodes = output_nodes.to(device)

            logits = model(g, input_nodes, blocks)
            y = labels[output_nodes].to(device)

            loss = loss_fn(logits, y.float())

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
            print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f}")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()

    world_size = args.world_size
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
