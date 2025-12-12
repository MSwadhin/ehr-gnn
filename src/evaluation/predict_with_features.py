# src/evaluation/predict_with_features.py

import os
import argparse

import numpy as np
import torch
import dgl
from dgl.dataloading import NeighborSampler, DataLoader

from src.configs import load_paths_config, load_training_config
from src.training.train_with_features import GraphSAGEFeat, GCNFeat  # reuse same classes


TASK_NAMES = ["visit_mort", "patient_mort", "visit_readm", "patient_readm"]


def _get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _load_split_ids(paths, split: str) -> torch.Tensor:
    fname = f"{split}_ids.txt"
    path = os.path.join(paths.data_splits_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{fname} not found at {path}")
    with open(path, "r") as f:
        ids = [int(x.strip()) for x in f if x.strip()]
    return torch.tensor(ids, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["graphsage", "gcn"],
        help="Feature-based model type used during training.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to predict on.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint. If None, infer from model name.",
    )
    args = parser.parse_args()

    device = _get_device()
    print(f"[FeatPredict] Using device: {device}")

    paths = load_paths_config()
    train_cfg = load_training_config()
    model_name = args.model.lower()
    split = args.split

    # ------------------------------------------------------------
    # 1. Load graph
    # ------------------------------------------------------------
    graph_path = os.path.join(paths.data_graph_dir, "dgl", "full_graph.bin")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"full_graph.bin not found at {graph_path}")
    glist, _ = dgl.load_graphs(graph_path)
    g = glist[0]
    num_nodes = g.num_nodes()
    print(f"Graph: {num_nodes} nodes, {g.num_edges()} edges")

    # ------------------------------------------------------------
    # 2. Load features
    # ------------------------------------------------------------
    feat_path = os.path.join(paths.data_graph_dir, "node_features.pt")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"node_features.pt not found at {feat_path}. "
            "Run src.features.build_node_features first."
        )
    feat_obj = torch.load(feat_path, map_location="cpu")
    feats = feat_obj["features"]  # [N, F]
    if feats.shape[0] != num_nodes:
        raise ValueError(
            f"features.shape[0] ({feats.shape[0]}) != num_nodes ({num_nodes})"
        )
    in_dim = feats.shape[1]
    print(f"Loaded node_features: shape={feats.shape}")
    g.ndata["feat"] = feats  # kept on CPU; DataLoader will move blocks to device

    # ------------------------------------------------------------
    # 3. Load labels/mask (for shape and mask alignment)
    # ------------------------------------------------------------
    labels_path = os.path.join(paths.data_graph_dir, "labels_outcomes.pt")
    mask_path = os.path.join(paths.data_graph_dir, "labels_outcomes_mask.pt")
    if not (os.path.exists(labels_path) and os.path.exists(mask_path)):
        raise FileNotFoundError("labels_outcomes.pt or labels_outcomes_mask.pt missing in data/graph")

    labels = torch.load(labels_path, map_location="cpu")        # [N, T]
    label_mask = torch.load(mask_path, map_location="cpu")      # [N, T]
    num_tasks = labels.shape[1]
    print(f"Num tasks (label dims): {num_tasks}")

    # ------------------------------------------------------------
    # 4. Build model and load checkpoint
    # ------------------------------------------------------------
    if args.checkpoint is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_dir = os.path.join(project_root, "models")
        ckpt_name = f"{model_name}_feat_mort_readm_allparts.pt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    else:
        ckpt_path = args.checkpoint

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")

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

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------
    # 5. Load split node IDs
    # ------------------------------------------------------------
    split_ids = _load_split_ids(paths, split)
    print(f"{split} split: {len(split_ids)} nodes")

    # ------------------------------------------------------------
    # 6. DataLoader and prediction loop
    # ------------------------------------------------------------
    fanouts = list(train_cfg.sampling.fanouts)
    print(f"NeighborSampler fanouts: {fanouts}")
    sampler = NeighborSampler(fanouts)

    dataloader = DataLoader(
        g,
        split_ids,
        sampler,
        batch_size=train_cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=train_cfg.sampling.num_workers,
        device=device,
    )

    probs = torch.zeros((num_nodes, num_tasks), dtype=torch.float32)
    pred_mask = torch.zeros((num_nodes, num_tasks), dtype=torch.bool)

    with torch.no_grad():
        for input_nodes, output_nodes, blocks in dataloader:
            x = g.ndata["feat"][input_nodes].to(device)
            logits = model(blocks, x)  # [B, T]
            p = torch.sigmoid(logits).cpu()  # probabilities

            probs[output_nodes] = p
            pred_mask[output_nodes] = True  # we predicted for these nodes

    # ------------------------------------------------------------
    # 7. Save prediction file
    # ------------------------------------------------------------
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preds_dir = os.path.join(project_root, "predictions")
    os.makedirs(preds_dir, exist_ok=True)

    out_name = f"{model_name}_feat_preds_{split}.pt"
    out_path = os.path.join(preds_dir, out_name)

    torch.save(
        {
            "model": f"{model_name}_feat",
            "split": split,
            "probs": probs,
            "mask": pred_mask,
            "task_names": TASK_NAMES[:num_tasks],
        },
        out_path,
    )

    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
