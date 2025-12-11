import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.configs import load_paths_config
from src.utils.io_utils import read_parquet, ensure_dir


def _build_node_mapping(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Build a mapping from original node_id (patient/clinical) to contiguous METIS ids.

    METIS ids are 1-based: 1..N
    """
    src_ids = edges["src_id"].to_numpy()
    dst_ids = edges["dst_id"].to_numpy()
    all_ids = np.concatenate([src_ids, dst_ids])
    unique_ids = np.unique(all_ids)

    # Original node_id -> metis_id (1..N)
    metis_ids = np.arange(1, len(unique_ids) + 1, dtype=np.int64)
    mapping_df = pd.DataFrame(
        {
            "node_id": unique_ids,
            "metis_id": metis_ids,
        }
    )
    return mapping_df


def _build_undirected_edges(
    edges: pd.DataFrame, mapping: pd.DataFrame
) -> Tuple[pd.DataFrame, int]:
    """
    Convert directed patient→clinical edges into undirected weighted edges
    for METIS, using the compressed metis_id space.

    Returns:
      undirected_edges: DataFrame with columns [u, v, weight]
      num_nodes: number of nodes in compressed graph
    """
    # Join to get metis src/dst
    edges_m = edges.merge(
        mapping, how="left", left_on="src_id", right_on="node_id"
    ).rename(columns={"metis_id": "u"})
    edges_m = edges_m.drop(columns=["node_id"])

    edges_m = edges_m.merge(
        mapping, how="left", left_on="dst_id", right_on="node_id"
    ).rename(columns={"metis_id": "v"})
    edges_m = edges_m.drop(columns=["node_id"])

    if edges_m["u"].isna().any() or edges_m["v"].isna().any():
        raise RuntimeError("Missing metis_id for some src/dst nodes after mapping merge.")

    edges_m["u"] = edges_m["u"].astype("int64")
    edges_m["v"] = edges_m["v"].astype("int64")

    # Canonicalize to undirected (u <= v) and group by to sum weights if multiple edges
    u = edges_m["u"].to_numpy()
    v = edges_m["v"].to_numpy()
    w = edges_m["weight"].to_numpy()

    u_min = np.minimum(u, v)
    v_max = np.maximum(u, v)

    undirected = pd.DataFrame(
        {
            "u": u_min,
            "v": v_max,
            "weight": w,
        }
    )

    undirected = (
        undirected
        .groupby(["u", "v"], as_index=False)
        .agg({"weight": "sum"})
    )

    num_nodes = mapping["metis_id"].max()
    return undirected, int(num_nodes)


def _build_adjacency(
    undirected_edges: pd.DataFrame,
    num_nodes: int,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build adjacency list for METIS from undirected edges.

    Returns dict: node -> list of (neighbor, weight)
    """
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(1, num_nodes + 1)}

    for _, row in undirected_edges.iterrows():
        u = int(row["u"])
        v = int(row["v"])
        w = float(row["weight"])

        if u == v:
            # Skip self-loops for METIS
            continue

        adj[u].append((v, w))
        adj[v].append((u, w))

    # Optionally sort neighbors for deterministic output
    for node, nbrs in adj.items():
        adj[node] = sorted(nbrs, key=lambda x: x[0])

    return adj


def export_to_metis() -> str:
    """
    Export contracted patient↔clinical graph to a METIS .graph file.

    Input:
      data/graph/contracted_edges.parquet
        columns: src_id, dst_id, src_type, dst_type, edge_type, weight

    Outputs:
      data/graph/metis/contracted.graph
      data/graph/metis/node_mapping.parquet
    """
    paths = load_paths_config()

    graph_dir = paths.data_graph_dir
    contracted_path = os.path.join(graph_dir, paths.contracted_edges_parquet)

    metis_dir = paths.metis_work_dir
    metis_graph_filename = paths.metis_graph_filename
    metis_graph_path = os.path.join(metis_dir, metis_graph_filename)
    node_map_path = os.path.join(metis_dir, "node_mapping.parquet")

    print(f"Loading contracted edges from {contracted_path}")
    edges = read_parquet(contracted_path)

    expected_cols = {"src_id", "dst_id", "weight"}
    missing = expected_cols - set(edges.columns)
    if missing:
        raise ValueError(f"contracted_edges.parquet missing columns: {missing}")

    # 1) Build node mapping
    print("Building node mapping (node_id -> metis_id)...")
    mapping_df = _build_node_mapping(edges)
    num_nodes = len(mapping_df)
    print(f"Compressed to {num_nodes} METIS nodes.")

    ensure_dir(node_map_path)
    print(f"Writing node mapping -> {node_map_path}")
    mapping_df.to_parquet(node_map_path, index=False)

    # 2) Build undirected edges in METIS id space
    print("Building undirected weighted edges...")
    undirected_edges, num_nodes_check = _build_undirected_edges(edges, mapping_df)
    assert num_nodes == num_nodes_check

    num_undirected_edges = len(undirected_edges)
    print(f"Number of undirected edges (unique u,v): {num_undirected_edges}")

    # 3) Build adjacency
    print("Building adjacency lists...")
    adj = _build_adjacency(undirected_edges, num_nodes=num_nodes)

    # 4) Write METIS .graph file
    ensure_dir(metis_graph_path)
    print(f"Writing METIS graph -> {metis_graph_path}")

    # Format: first line: <num_nodes> <num_edges> <fmt>
    # fmt=1 => edge weights
    # num_edges is number of undirected edges
    with open(metis_graph_path, "w") as f:
        header = f"{num_nodes} {num_undirected_edges} 1\n"
        f.write(header)

        for node in range(1, num_nodes + 1):
            nbrs = adj.get(node, [])
            if not nbrs:
                f.write("\n")
                continue
            parts = []
            for (v, w) in nbrs:
                # METIS expects integer weights; you can scale if you want finer granularity.
                w_int = int(round(w))
                if w_int <= 0:
                    w_int = 1
                parts.append(f"{v} {w_int}")
            line = " ".join(parts) + "\n"
            f.write(line)

    print("METIS export complete.")
    return metis_graph_path


if __name__ == "__main__":
    export_to_metis()
