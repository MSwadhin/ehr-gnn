import os

import pandas as pd

from src.configs import load_paths_config, load_semantic_graph_config
from src.utils.io_utils import read_parquet, write_parquet, ensure_dir


def build_nodes_and_remap_edges() -> tuple[str, str]:
    """
    Build global node IDs from the heterogeneous edge list and remap edges to use IDs.

    Input:
      data/processed/ehr_edges.parquet
        columns: src_type, src_key, dst_type, dst_key, edge_type, weight

    Output:
      data/graph/nodes.parquet
        columns: node_id, node_type, raw_key

      data/graph/ehr_edges.parquet
        columns: src_id, dst_id, src_type, dst_type, edge_type, weight
    """
    paths = load_paths_config()
    _ = load_semantic_graph_config()  # not strictly needed here, but available if you want sanity checks

    processed_edges_path = os.path.join(paths.data_processed_dir, "ehr_edges.parquet")
    graph_dir = paths.data_graph_dir

    nodes_out_path = os.path.join(graph_dir, paths.nodes_parquet)
    graph_edges_out_path = os.path.join(graph_dir, paths.ehr_edges_parquet)

    print(f"Loading edge list from {processed_edges_path}")
    edges = read_parquet(processed_edges_path)

    expected_cols = {"src_type", "src_key", "dst_type", "dst_key", "edge_type", "weight"}
    missing = expected_cols - set(edges.columns)
    if missing:
        raise ValueError(f"ehr_edges.parquet missing columns: {missing}")

    # ----------------------------------------------------------------------
    # 1) Build node table: unique (node_type, raw_key)
    # ----------------------------------------------------------------------
    print("Constructing unique node list...")

    src_nodes = (
        edges[["src_type", "src_key"]]
        .rename(columns={"src_type": "node_type", "src_key": "raw_key"})
    )
    dst_nodes = (
        edges[["dst_type", "dst_key"]]
        .rename(columns={"dst_type": "node_type", "dst_key": "raw_key"})
    )

    all_nodes = pd.concat([src_nodes, dst_nodes], axis=0, ignore_index=True)
    all_nodes = all_nodes.drop_duplicates().reset_index(drop=True)

    # Assign global node IDs
    all_nodes["node_id"] = all_nodes.index.astype("int64")

    # Reorder columns
    nodes_df = all_nodes[["node_id", "node_type", "raw_key"]]

    print(f"Total unique nodes: {len(nodes_df)}")

    # ----------------------------------------------------------------------
    # 2) Create a mapping (node_type, raw_key) -> node_id
    # ----------------------------------------------------------------------
    print("Building node_id mapping and remapping edges...")

    # We'll merge instead of building a Python dict to avoid memory blowups.
    # First, map src
    edges_src = edges.merge(
        nodes_df,
        how="left",
        left_on=["src_type", "src_key"],
        right_on=["node_type", "raw_key"],
        suffixes=("", "_node_src"),
    ).rename(columns={"node_id": "src_id"})

    # Drop helper columns from merge
    edges_src = edges_src.drop(columns=["node_type", "raw_key"])

    # Now map dst
    edges_mapped = edges_src.merge(
        nodes_df,
        how="left",
        left_on=["dst_type", "dst_key"],
        right_on=["node_type", "raw_key"],
        suffixes=("", "_node_dst"),
    ).rename(columns={"node_id": "dst_id"})

    edges_mapped = edges_mapped.drop(columns=["node_type", "raw_key"])

    # Sanity check: no unmapped IDs
    if edges_mapped["src_id"].isna().any() or edges_mapped["dst_id"].isna().any():
        n_bad_src = edges_mapped["src_id"].isna().sum()
        n_bad_dst = edges_mapped["dst_id"].isna().sum()
        raise RuntimeError(
            f"Found unmapped node IDs: {n_bad_src} src_id missing, {n_bad_dst} dst_id missing."
        )

    # Reorder columns in final edge table
    edges_final = edges_mapped[
        ["src_id", "dst_id", "src_type", "dst_type", "edge_type", "weight"]
    ].copy()

    # Cast IDs to int64
    edges_final["src_id"] = edges_final["src_id"].astype("int64")
    edges_final["dst_id"] = edges_final["dst_id"].astype("int64")

    print(f"Total edges (after remap): {len(edges_final)}")

    # ----------------------------------------------------------------------
    # 3) Write outputs
    # ----------------------------------------------------------------------
    ensure_dir(nodes_out_path)
    ensure_dir(graph_edges_out_path)

    print(f"Writing nodes -> {nodes_out_path}")
    write_parquet(nodes_df, nodes_out_path)

    print(f"Writing remapped edges -> {graph_edges_out_path}")
    write_parquet(edges_final, graph_edges_out_path)

    return nodes_out_path, graph_edges_out_path


if __name__ == "__main__":
    build_nodes_and_remap_edges()
