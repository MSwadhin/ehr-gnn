# src/training/train_with_features.py

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import dgl
from dgl.dataloading import NeighborSampler, DataLoader
from dgl.nn import SAGEConv, GraphConv

from src.configs import load_paths_config, load_training_config


# -------------------------------------------------------------------
# Feature-based models (self-contained, do not depend on src.models.*)
# -------------------------------------------------------------------

class GraphSAGEFeat(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggregator_type: str = "mean",
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            SAGEConv(in_feats, hidden_feats, aggregator_type=aggregator_type)
        )
        for _ in range(num_layers - 2):
            self.layers.append(
                SAGEConv(hidden_feats, hidden_feats, aggregator_type=aggregator_type)
            )
        if num_layers > 1:
            self.layers.append(
                SAGEConv(hidden_feats, out_feats, aggregator_type=aggregator_type)
            )
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, blocks, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(blocks[i], h)
            if i != len(self.layers) - 1:
                h = self.act(h)
                h = self.dropout(h)
        return h


class GCNFeat(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        allow_zero_in_degree: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            GraphConv(
                in_feats,
                hidden_feats,
                norm="both",
                weight=True,
                bias=True,
                allow_zero_in_degree=allow_zero_in_degree,
            )
        )
        for _ in range(num_layers - 2):
            self.layers.append(
                GraphConv(
                    hidden_feats,
                    hidden_feats,
                    norm="both",
                    weight=True,
                    bias=True,
                    allow_zero_in_degree=allow_zero_in_degree,
                )
            )
        if num_layers > 1:
            self.layers.append(
                GraphConv(
                    hidden_feats,
                    out_feats,
                    norm="both",
                    weight=True,
                    bias=True,
                    allow_zero_in_degree=allow_zero_in_degree,
                )
            )
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, blocks, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(blocks[i], h)
            if i != len(self.layers) - 1:
                h = self.act(h)
                h = self.dropout(h)
        return h


# -------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------

def _get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _compute_pos_weight(labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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


# -------------------------------------------------------------------
# Main training logic
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="graphsage",
        choices=["graphsage", "gcn"],
        help="Feature-based model type to train.",
    )
    args = parser.parse_args()

    device = _get_device()
    print(f"[FeatTrainer] Using device: {device}")

    paths = load_paths_config()
    train_cfg = load_training_config()
    model_name = args.model.lower()
    print(f"Training feature-based model: {model_name}")

    # ---------------------------
    # 1. Load graph
    # ---------------------------
    graph_path = os.path.join(paths.data_graph_dir, "dgl", "full_graph.bin")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"full_graph.bin not found at {graph_path}")

    glist, _ = dgl.load_graphs(graph_path)
    g = glist[0]
    num_nodes = g.num_nodes()
    print(f"Graph: {num_nodes} nodes, {g.num_edges()} edges")

    # ---------------------------
    # 2. Load features
    # ---------------------------
    feat_path = os.path.join(paths.data_graph_dir, "node_features.pt")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"node_features.pt not found at {feat_path}. "
            "Run src.features.build_node_features first."
        )

    feat_obj = torch.load(feat_path, map_location="cpu")
    feats = feat_obj["features"]
    if feats.shape[0] != num_nodes:
        raise ValueError(
            f"features shape[0] ({feats.shape[0]}) != num_nodes ({num_nodes})"
        )

    in_dim = feats.shape[1]
    print(f"Loaded node features: shape={feats.shape}")

    g.ndata["feat"] = feats  # keep on CPU; DataLoader will move to device

    # ---------------------------
    # 3. Load labels/mask + train_ids
    # ---------------------------
    labels_path = os.path.join(paths.data_graph_dir, "labels_outcomes.pt")
    mask_path = os.path.join(paths.data_graph_dir, "labels_outcomes_mask.pt")
    train_ids_path = os.path.join(paths.data_splits_dir, "train_ids.txt")

    if not (os.path.exists(labels_path) and os.path.exists(mask_path) and os.path.exists(train_ids_path)):
        raise FileNotFoundError("labels_outcomes / labels_outcomes_mask / train_ids missing.")

    labels = torch.load(labels_path, map_location="cpu")        # [N, T]
    label_mask = torch.load(mask_path, map_location="cpu")      # [N, T]

    with open(train_ids_path, "r") as f:
        train_ids = [int(line.strip()) for line in f if line.strip()]
    train_ids = torch.tensor(train_ids, dtype=torch.long)
    num_tasks = labels.shape[1]

    print(f"Num tasks: {num_tasks}, train_ids: {len(train_ids)}")

    # ---------------------------
    # 4. pos_weight + loss
    # ---------------------------
    pos_weight = _compute_pos_weight(labels, label_mask)
    print("pos_weight per task:", pos_weight.tolist())

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction="none")

    # ---------------------------
    # 5. Build model + optimizer
    # ---------------------------
    if model_name == "graphsage":
        model = GraphSAGEFeat(
            in_feats=in_dim,
            hidden_feats=train_cfg.model.hidden_dim,
            out_feats=num_tasks,
            num_layers=train_cfg.model.num_layers,
            dropout=train_cfg.model.dropout,
            aggregator_type=getattr(train_cfg.model, "aggregator", "mean"),
        )
    else:
        model = GCNFeat(
            in_feats=in_dim,
            hidden_feats=train_cfg.model.hidden_dim,
            out_feats=num_tasks,
            num_layers=train_cfg.model.num_layers,
            dropout=train_cfg.model.dropout,
            allow_zero_in_degree=True,
        )

    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg.training.lr,
        weight_decay=train_cfg.training.weight_decay,
    )

    # ---------------------------
    # 6. DataLoader with NeighborSampler
    # ---------------------------
    fanouts = list(train_cfg.sampling.fanouts)
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
        device=device,
    )

    # ---------------------------
    # 7. Training loop
    # ---------------------------
    num_epochs = train_cfg.training.num_epochs
    print(f"Starting feature-based training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for input_nodes, output_nodes, blocks in dataloader:
            x = g.ndata["feat"][input_nodes].to(device)
            logits = model(blocks, x)  # [B, num_tasks]

            y = labels[output_nodes].to(device)
            m = label_mask[output_nodes].to(device)

            raw_loss = loss_fn(logits, y)   # [B, T]
            weighted = raw_loss * m

            denom = m.sum().clamp(min=1.0)
            loss = weighted.sum() / denom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        print(f"[Epoch {epoch}] model={model_name}-feat avg_loss={avg_loss:.4f}")

    print("Feature-based training finished.")

    # ---------------------------
    # 8. Save checkpoint
    # ---------------------------
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(project_root, "models")
    os.makedirs(ckpt_dir, exist_ok=True)

    out_ckpt = os.path.join(ckpt_dir, f"{model_name}_feat_mort_readm_allparts.pt")
    torch.save(model.state_dict(), out_ckpt)
    print(f"Saved feature-based model checkpoint to: {out_ckpt}")


if __name__ == "__main__":
    main()
