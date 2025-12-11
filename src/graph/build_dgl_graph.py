import os
from typing import Dict

import torch
import dgl
import pandas as pd

from src.configs import load_paths_config, load_semantic_graph_config
from src.utils.io_utils import read_parquet, ensure_dir, write_parquet


NODE_TYPE_TO_INT = {
    "patient": 0,
    "visit": 1,
    "diagnosis": 2,
    "procedure": 3,
    "medication": 4,
    "lab_bucket": 5,
}


def build_dgl_graph() -> str:
    """
    Build a homogeneous DGLGraph from nodes_with_partitions + ehr_edges.

    Inputs:
      data/graph/nodes_with_partitions.parquet
        columns: node_id, node_type, raw_key, partition

      data/graph/ehr_edges.parquet
        columns: src_id, dst_id, src_type, dst_type, edge_type, weight

    Outputs:
      data/graph/dgl/full_graph.bin   (DGL graph + node features)
      data/graph/dgl/nodes_meta.parquet   (node_id, node_type, partition)
    """
    paths = load_paths_config()
    _ = load_semantic_graph_config()

    graph_dir = paths.data_graph_dir
    nodes_path = os.path.join(graph_dir, "nodes_with_partitions.parquet")
    edges_path = os.path.join(graph_dir, paths.ehr_edges_parquet)

    dgl_dir = os.path.join(graph_dir, "dgl")
    graph_out_path = os.path.join(dgl_dir, "full_graph.bin")
    nodes_meta_out = os.path.join(dgl_dir, "nodes_meta.parquet")

    print(f"Loading nodes_with_partitions from {nodes_path}")
    nodes = read_parquet(nodes_path)
    print(f"Loading edges from {edges_path}")
    edges = read_parquet(edges_path)

    # Basic checks
    for col in ["node_id", "node_type", "partition"]:
        if col not in nodes.columns:
            raise ValueError(f"nodes_with_partitions missing column: {col}")

    needed_edge_cols = {"src_id", "dst_id", "edge_type", "weight"}
    missing = needed_edge_cols - set(edges.columns)
    if missing:
        raise ValueError(f"ehr_edges.parquet missing columns: {missing}")

    # Ensure node_id is contiguous from 0..N-1
    num_nodes = nodes["node_id"].max() + 1
    if len(nodes) != num_nodes:
        # If there are gaps, reindex – but in your pipeline there shouldn't be any.
        raise RuntimeError(
            f"Expected node_id to be contiguous 0..N-1, got max={nodes['node_id'].max()} "
            f"but len(nodes)={len(nodes)}."
        )

    print(f"Building DGLGraph with {num_nodes} nodes and {len(edges)} edges")

    # src/dst as tensors
    src = torch.from_numpy(edges["src_id"].astype("int64").to_numpy())
    dst = torch.from_numpy(edges["dst_id"].astype("int64").to_numpy())

    g = dgl.graph((src, dst), num_nodes=num_nodes)

    # Node features: partition, node_type (int), raw ID as embedding index
    part = torch.from_numpy(nodes["partition"].astype("int64").to_numpy())
    g.ndata["partition"] = part

    # Encode node_type as int
    ntype_int = nodes["node_type"].map(
        lambda x: NODE_TYPE_TO_INT.get(x, -1)
    ).astype("int64")
    g.ndata["ntype"] = torch.from_numpy(ntype_int.to_numpy())

    # Use node_id itself as an index into an embedding
    g.ndata["feat_id"] = torch.arange(num_nodes, dtype=torch.long)

    # Edge weight as feature (for possible weighted aggregation)
    g.edata["weight"] = torch.from_numpy(edges["weight"].astype("float32").to_numpy())

    # Persist DGL graph
    ensure_dir(graph_out_path)
    dgl.save_graphs(graph_out_path, g)
    print(f"Saved DGL graph -> {graph_out_path}")

    # Save meta (for debugging/analysis)
    meta_df = nodes[["node_id", "node_type", "partition"]].copy()
    ensure_dir(nodes_meta_out)
    write_parquet(meta_df, nodes_meta_out)
    print(f"Saved node metadata -> {nodes_meta_out}")

    return graph_out_path


if __name__ == "__main__":
    build_dgl_graph()
