import os

import pandas as pd

from src.configs import load_paths_config, load_semantic_graph_config
from src.utils.io_utils import read_parquet


def analyze_partitions() -> None:
    """
    Analyze semantic partitions:
      - node counts per partition (by node_type)
      - unweighted edge-cut ratio
      - weighted edge-cut ratio (overall and per edge_type)
    """
    paths = load_paths_config()
    sem_cfg = load_semantic_graph_config()

    graph_dir = paths.data_graph_dir

    nodes_path = os.path.join(graph_dir, "nodes_with_partitions.parquet")
    edges_path = os.path.join(graph_dir, "ehr_edges.parquet")

    print(f"Loading nodes_with_partitions from {nodes_path}")
    nodes = read_parquet(nodes_path)

    if "partition" not in nodes.columns:
        raise ValueError("nodes_with_partitions.parquet must contain a 'partition' column.")

    print(f"Loading edges from {edges_path}")
    edges = read_parquet(edges_path)

    required_edge_cols = {"src_id", "dst_id", "edge_type", "weight"}
    missing = required_edge_cols - set(edges.columns)
    if missing:
        raise ValueError(f"ehr_edges.parquet missing columns: {missing}")

    # ----------------------------------------------------------------------
    # Node stats
    # ----------------------------------------------------------------------
    print("\n=== Node partition distribution ===")
    part_counts = nodes["partition"].value_counts().sort_index()
    print(part_counts)

    print("\n=== Node partition distribution by node_type ===")
    nt_part = (
        nodes.groupby(["node_type", "partition"])
        .size()
        .reset_index(name="count")
        .sort_values(["node_type", "partition"])
    )
    print(nt_part.head(50))

    # ----------------------------------------------------------------------
    # Edge cut stats
    # ----------------------------------------------------------------------
    print("\n=== Edge cut analysis ===")

    node_part = nodes[["node_id", "partition"]].rename(columns={"node_id": "src_id", "partition": "src_part"})
    edges = edges.merge(node_part, on="src_id", how="left")

    node_part_dst = nodes[["node_id", "partition"]].rename(columns={"node_id": "dst_id", "partition": "dst_part"})
    edges = edges.merge(node_part_dst, on="dst_id", how="left")

    # Drop edges where partition missing on either side
    edges = edges.dropna(subset=["src_part", "dst_part"]).copy()
    edges["src_part"] = edges["src_part"].astype("int64")
    edges["dst_part"] = edges["dst_part"].astype("int64")

    total_edges = len(edges)
    total_weight = edges["weight"].sum()

    edges["is_cut"] = edges["src_part"] != edges["dst_part"]

    cut_edges = edges[edges["is_cut"]]
    num_cut = len(cut_edges)
    cut_weight = cut_edges["weight"].sum()

    print(f"Total edges considered: {total_edges}")
    print(f"Unweighted cut edges: {num_cut} ({num_cut / total_edges:.4f} of edges)")
    print(f"Total edge weight: {total_weight:.2f}")
    print(f"Weighted cut: {cut_weight:.2f} ({cut_weight / total_weight:.4f} of weight)")

    # Per edge_type
    print("\n=== Edge cut by edge_type ===")
    stats = []

    for etype in edges["edge_type"].unique():
        sub = edges[edges["edge_type"] == etype]
        if sub.empty:
            continue
        total_e = len(sub)
        total_w = sub["weight"].sum()
        cut_sub = sub[sub["is_cut"]]
        num_cut_e = len(cut_sub)
        cut_w = cut_sub["weight"].sum()
        stats.append(
            {
                "edge_type": etype,
                "total_edges": total_e,
                "cut_edges": num_cut_e,
                "cut_ratio_edges": num_cut_e / total_e,
                "total_weight": total_w,
                "cut_weight": cut_w,
                "cut_ratio_weight": cut_w / total_w if total_w > 0 else 0.0,
            }
        )

    stats_df = pd.DataFrame(stats).sort_values("edge_type")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    analyze_partitions()
