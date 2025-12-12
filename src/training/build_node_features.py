# src/features/build_node_features.py

import os
import numpy as np
import torch

from src.configs import load_paths_config
from src.utils.io_utils import read_parquet


def main():
    paths = load_paths_config()

    nodes_path = os.path.join(paths.data_graph_dir, "nodes.parquet")
    edges_path = os.path.join(paths.data_graph_dir, "ehr_edges.parquet")

    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        raise FileNotFoundError(f"Missing nodes/edges parquet in {paths.data_graph_dir}")

    print(f"Loading nodes from {nodes_path}")
    nodes = read_parquet(nodes_path)  # expects columns: node_id, node_type, raw_key

    print(f"Loading edges from {edges_path}")
    edges = read_parquet(edges_path)  # expects: src_id, dst_id, src_type, dst_type, edge_type, weight

    num_nodes = int(nodes["node_id"].max()) + 1
    print(f"num_nodes={num_nodes}")

    # -------------------------------------------------------
    # 1) Node type one-hot
    # -------------------------------------------------------
    node_types = sorted(nodes["node_type"].unique().tolist())
    type_to_idx = {t: i for i, t in enumerate(node_types)}
    num_types = len(node_types)
    print(f"Found node_types: {node_types}")

    type_idx = nodes["node_type"].map(type_to_idx).to_numpy()
    node_ids = nodes["node_id"].to_numpy()

    type_onehot = np.zeros((num_nodes, num_types), dtype=np.float32)
    type_onehot[node_ids, type_idx] = 1.0

    # -------------------------------------------------------
    # 2) Total degree (undirected)
    # -------------------------------------------------------
    src = edges["src_id"].to_numpy()
    dst = edges["dst_id"].to_numpy()

    all_ends = np.concatenate([src, dst], axis=0)
    deg = np.bincount(all_ends, minlength=num_nodes).astype(np.float32)
    deg_feat = np.log1p(deg).reshape(-1, 1)  # shape [N, 1]

    # -------------------------------------------------------
    # 3) Neighbor-type counts per node
    #    For each node, how many neighbors of each node_type?
    # -------------------------------------------------------
    src_type = edges["src_type"].to_numpy()
    dst_type = edges["dst_type"].to_numpy()

    # Map types to small ints
    src_type_idx = np.vectorize(type_to_idx.get)(src_type)
    dst_type_idx = np.vectorize(type_to_idx.get)(dst_type)

    # Build contributions: (node_id, neighbor_type_idx)
    node_ids_neigh = np.concatenate([src, dst], axis=0)
    neigh_types = np.concatenate([dst_type_idx, src_type_idx], axis=0)

    neighbor_counts = np.zeros((num_nodes, num_types), dtype=np.float32)
    # equivalent to:
    # for (n, t) in zip(node_ids_neigh, neigh_types): neighbor_counts[n, t] += 1
    np.add.at(neighbor_counts, (node_ids_neigh, neigh_types), 1)

    neighbor_counts_feat = np.log1p(neighbor_counts)  # [N, num_types]

    # -------------------------------------------------------
    # 4) Concatenate all features
    #    [ node_type_onehot | log(1+degree) | log(1+neighbor_counts_by_type) ]
    # -------------------------------------------------------
    feat = np.concatenate([type_onehot, deg_feat, neighbor_counts_feat], axis=1)
    feat_t = torch.from_numpy(feat)

    out_path = os.path.join(paths.data_graph_dir, "node_features.pt")
    torch.save(
        {
            "features": feat_t,
            "node_types": node_types,
            "description": "node_type one-hot + log(1+degree) + log(1+neighbor_counts_by_type)",
        },
        out_path,
    )

    print(f"Saved node_features to {out_path}")
    print(f"Feature shape: {feat_t.shape} (num_nodes x feature_dim)")


if __name__ == "__main__":
    main()
