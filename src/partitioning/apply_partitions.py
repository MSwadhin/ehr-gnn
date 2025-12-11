import os
from typing import List

import numpy as np
import pandas as pd

from src.configs import load_paths_config, load_semantic_graph_config
from src.utils.io_utils import read_parquet, write_parquet, ensure_dir


def _load_metis_partitions(part_file: str) -> np.ndarray:
    """
    Load gpmetis partition file: one integer per line, partition id in [0, k-1].
    Line i (0-based) corresponds to METIS node id i+1.
    """
    parts: List[int] = []
    with open(part_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts.append(int(line))
    return np.array(parts, dtype=np.int64)


def apply_partitions_to_nodes() -> str:
    """
    Map METIS partitions back to original node_ids and propagate to visits.

    Inputs:
      data/graph/metis/node_mapping.parquet
        columns: node_id, metis_id

      data/graph/metis/contracted.graph.part.<K>

      data/graph/nodes.parquet
        columns: node_id, node_type, raw_key

      data/graph/ehr_edges.parquet
        columns: src_id, dst_id, src_type, dst_type, edge_type, weight

    Outputs:
      data/graph/nodes_with_partitions.parquet
        columns: node_id, node_type, raw_key, partition
    """
    paths = load_paths_config()
    sem_cfg = load_semantic_graph_config()

    graph_dir = paths.data_graph_dir
    metis_dir = paths.metis_work_dir
    graph_filename = paths.metis_graph_filename
    num_parts = paths.metis_num_parts

    node_map_path = os.path.join(metis_dir, "node_mapping.parquet")
    part_file = os.path.join(metis_dir, f"{graph_filename}.part.{num_parts}")

    if not os.path.exists(node_map_path):
        raise FileNotFoundError(f"Node mapping not found at {node_map_path}. Run export_to_metis first.")
    if not os.path.exists(part_file):
        raise FileNotFoundError(f"Partition file not found at {part_file}. Run gpmetis first.")

    print(f"Loading node_mapping from {node_map_path}")
    node_map = read_parquet(node_map_path)  # node_id, metis_id
    if not {"node_id", "metis_id"}.issubset(node_map.columns):
        raise ValueError("node_mapping.parquet must have columns: node_id, metis_id")

    print(f"Loading METIS partitions from {part_file}")
    parts = _load_metis_partitions(part_file)

    if len(parts) != len(node_map):
        raise RuntimeError(
            f"Partition file length ({len(parts)}) does not match number of METIS nodes ({len(node_map)})."
        )

    # Build metis_id -> partition mapping
    part_df = pd.DataFrame(
        {
            "metis_id": node_map["metis_id"].to_numpy(),
            "partition": parts,
        }
    )

    # Merge: node_id + metis_id + partition
    node_part = node_map.merge(part_df, on="metis_id", how="left")
    if node_part["partition"].isna().any():
        raise RuntimeError("Some metis_ids did not get a partition assigned.")

    node_part = node_part[["node_id", "partition"]].copy()
    node_part["node_id"] = node_part["node_id"].astype("int64")
    node_part["partition"] = node_part["partition"].astype("int64")

    # Load full node table
    nodes_path = os.path.join(graph_dir, "nodes.parquet")
    print(f"Loading nodes from {nodes_path}")
    nodes = read_parquet(nodes_path)

    if not {"node_id", "node_type", "raw_key"}.issubset(nodes.columns):
        raise ValueError("nodes.parquet must have columns: node_id, node_type, raw_key")

    # Left join so only nodes present in METIS mapping get partitions initially
    nodes = nodes.merge(node_part, on="node_id", how="left")

    # At this point, patient + clinical nodes have partitions, visits do NOT.
    # Propagate patient partition to visit nodes via patient-visit edges.
    edges_path = os.path.join(graph_dir, "ehr_edges.parquet")
    print(f"Loading edges for partition propagation from {edges_path}")
    edges = read_parquet(edges_path)

    etypes = sem_cfg.edge_types
    et_patient_visit = etypes["patient_visit"]

    # patient-visit edges: src_type='patient', dst_type='visit'
    pv_edges = edges[
        (edges["edge_type"] == et_patient_visit)
        & (edges["src_type"] == "patient")
        & (edges["dst_type"] == "visit")
    ][["src_id", "dst_id"]].rename(columns={"src_id": "patient_id", "dst_id": "visit_id"})

    # Get patient partitions
    patient_part = nodes[nodes["node_type"] == "patient"][["node_id", "partition"]].rename(
        columns={"node_id": "patient_id", "partition": "patient_partition"}
    )

    pv_with_part = pv_edges.merge(patient_part, on="patient_id", how="left")

    # Some patients might not have a partition (should not happen if contraction used all patient nodes),
    # but we guard anyway.
    pv_with_part = pv_with_part.dropna(subset=["patient_partition"])

    visit_part = (
        pv_with_part[["visit_id", "patient_partition"]]
        .drop_duplicates(subset=["visit_id"])
        .rename(columns={"visit_id": "node_id", "patient_partition": "visit_partition"})
    )

    # Merge visit_partition into nodes where node_type == 'visit'
    nodes = nodes.merge(visit_part, on="node_id", how="left")

    # Prefer explicit partition (from METIS) when present; otherwise use visit_partition (for visits)
    def _choose_partition(row):
        if pd.notna(row["partition"]):
            return row["partition"]
        if pd.notna(row["visit_partition"]):
            return row["visit_partition"]
        return -1  # unknown / unassigned

    nodes["final_partition"] = nodes.apply(_choose_partition, axis=1).astype("int64")
    nodes = nodes.drop(columns=["partition", "visit_partition"])
    nodes = nodes.rename(columns={"final_partition": "partition"})

    # Write out updated node table with partitions
    out_path = os.path.join(graph_dir, "nodes_with_partitions.parquet")
    ensure_dir(out_path)
    print(f"Writing nodes with partitions -> {out_path}")
    write_parquet(nodes, out_path)

    return out_path


if __name__ == "__main__":
    apply_partitions_to_nodes()
