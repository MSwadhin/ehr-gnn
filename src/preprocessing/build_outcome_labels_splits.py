import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.configs import load_paths_config
from src.utils.io_utils import read_parquet, ensure_dir


RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2


def _split_patients(
    subject_ids: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split patient SUBJECT_ID_str into train/val/test.
    """
    rnd = random.Random(RANDOM_SEED)
    shuffled = subject_ids[:]
    rnd.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_subj = shuffled[:n_train]
    val_subj = shuffled[n_train : n_train + n_val]
    test_subj = shuffled[n_train + n_val :]

    return train_subj, val_subj, test_subj


def _compute_visit_readmission_flags(adm: pd.DataFrame, days: int = 30) -> Dict[str, int]:
    """
    For each admission of a patient, mark whether it has a 30-day readmission
    (index visit). Returns dict: HADM_ID_str -> 0/1.
    """
    adm = adm.copy()
    adm["SUBJECT_ID_str"] = adm["SUBJECT_ID"].astype(str)
    adm["HADM_ID_str"] = adm["HADM_ID"].astype(str)

    if not np.issubdtype(adm["ADMITTIME"].dtype, np.datetime64) or not np.issubdtype(
        adm["DISCHTIME"].dtype, np.datetime64
    ):
        raise ValueError("ADMITTIME and DISCHTIME must be datetime columns in admissions.parquet")

    flags: Dict[str, int] = {}

    # per-patient processing
    for subj, group in adm.groupby("SUBJECT_ID_str"):
        g = group.sort_values("ADMITTIME").reset_index(drop=True)

        hadm_ids = g["HADM_ID_str"].tolist()
        admit_times = g["ADMITTIME"].to_numpy()
        disch_times = g["DISCHTIME"].to_numpy()

        n = len(g)
        # default 0
        for h in hadm_ids:
            flags[h] = 0

        for i in range(n):
            disch_i = disch_times[i]
            hadm_i = hadm_ids[i]
            has_readm = False

            for j in range(i + 1, n):
                admit_j = admit_times[j]
                # numpy.datetime64 subtraction -> numpy.timedelta64
                delta = admit_j - disch_i
                # convert to days as float
                delta_days = float(delta / np.timedelta64(1, "D"))

                if delta_days <= 0:
                    # overlapping or same time; not a readmission
                    continue
                if delta_days <= days:
                    has_readm = True
                    break
                else:
                    # later admits are even further in time; no need to continue
                    break

            flags[hadm_i] = 1 if has_readm else 0

    return flags



def _attach_mortality_flag(adm: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'mort_flag' column to admissions:

    - If HOSPITAL_EXPIRE_FLAG / hospital_expire_flag exists, use that (0/1).
    - Else if DEATHTIME / deathtime exists, mort_flag = 1 if not null, else 0.
    - Else: error.
    """
    adm = adm.copy()
    cols_lower = {c.lower(): c for c in adm.columns}

    mort_series = None

    if "hospital_expire_flag" in cols_lower:
        col = cols_lower["hospital_expire_flag"]
        mort_series = adm[col].fillna(0).astype(int)
    elif "deathtime" in cols_lower:
        col = cols_lower["deathtime"]
        mort_series = adm[col].notna().astype(int)
    else:
        raise ValueError(
            "Admissions must contain either HOSPITAL_EXPIRE_FLAG/hospital_expire_flag "
            "or DEATHTIME/deathtime to derive mortality."
        )

    adm["mort_flag"] = mort_series.astype(int)
    return adm


def build_outcome_labels_and_splits() -> Tuple[str, str, str, str, str]:
    """
    Build BOTH visit-level and patient-level outcomes:
      - Visit mortality          -> task 0
      - Patient mortality        -> task 1
      - Visit 30-day readmission -> task 2
      - Patient 30-day readm     -> task 3

    Splits are patient-based:
      - Split SUBJECT_ID_str into train/val/test.
      - Each patient's patient node + visit nodes go to the same split.

    Inputs:
      data/processed/admissions.parquet
      data/graph/nodes_with_partitions.parquet

    Outputs:
      data/graph/labels_outcomes.pt          (torch.Tensor [num_nodes, 4])
      data/graph/labels_outcomes_mask.pt     (torch.Tensor [num_nodes, 4])
      data/splits/train_ids.txt              (node_ids: patients + visits)
      data/splits/val_ids.txt
      data/splits/test_ids.txt
    """
    paths = load_paths_config()

    proc_dir = paths.data_processed_dir
    graph_dir = paths.data_graph_dir
    splits_dir = paths.data_splits_dir

    admissions_path = os.path.join(proc_dir, "admissions.parquet")
    nodes_path = os.path.join(graph_dir, "nodes_with_partitions.parquet")

    labels_out_path = os.path.join(graph_dir, "labels_outcomes.pt")
    mask_out_path = os.path.join(graph_dir, "labels_outcomes_mask.pt")
    train_ids_path = os.path.join(splits_dir, "train_ids.txt")
    val_ids_path = os.path.join(splits_dir, "val_ids.txt")
    test_ids_path = os.path.join(splits_dir, "test_ids.txt")

    print(f"Loading admissions from {admissions_path}")
    admissions = read_parquet(admissions_path)
    print(f"Loading nodes_with_partitions from {nodes_path}")
    nodes = read_parquet(nodes_path)

    # ------------------------------------------------------------------
    # Required core columns
    # ------------------------------------------------------------------
    required_cols = {"SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"}
    missing = required_cols - set(admissions.columns)
    if missing:
        raise ValueError(f"ADMISSIONS missing columns: {missing}")

    # Attach mortality flag robustly
    admissions = _attach_mortality_flag(admissions)

    admissions = admissions.copy()
    admissions["SUBJECT_ID_str"] = admissions["SUBJECT_ID"].astype(str)
    admissions["HADM_ID_str"] = admissions["HADM_ID"].astype(str)

    # ------------------------------------------------------------------
    # Visit-level mortality labels (HADM_ID-level); use mort_flag
    # ------------------------------------------------------------------
    visit_mort = (
        admissions.groupby("HADM_ID_str")["mort_flag"]
        .max()
        .astype(int)
    )
    hadm_to_visit_mort: Dict[str, int] = visit_mort.to_dict()

    # ------------------------------------------------------------------
    # Patient-level mortality labels (SUBJECT_ID-level)
    # ------------------------------------------------------------------
    patient_mort = (
        admissions.groupby("SUBJECT_ID_str")["mort_flag"]
        .max()
        .astype(int)
    )
    subj_to_patient_mort: Dict[str, int] = patient_mort.to_dict()

    # ------------------------------------------------------------------
    # Visit-level readmission flags
    # ------------------------------------------------------------------
    print("Computing visit-level 30-day readmission flags...")
    hadm_to_readm = _compute_visit_readmission_flags(admissions, days=30)

    # ------------------------------------------------------------------
    # Patient-level readmission
    # ------------------------------------------------------------------
    subj_to_patient_readm: Dict[str, int] = {}
    for subj, group in admissions.groupby("SUBJECT_ID_str"):
        hadm_ids = group["HADM_ID_str"].tolist()
        flag = 0
        for h in hadm_ids:
            if hadm_to_readm.get(h, 0) == 1:
                flag = 1
                break
        subj_to_patient_readm[subj] = flag

    # ------------------------------------------------------------------
    # Map node_ids to SUBJECT_ID / HADM_ID
    # ------------------------------------------------------------------
    patient_nodes = nodes[nodes["node_type"] == "patient"][["node_id", "raw_key"]].copy()
    patient_nodes["SUBJECT_ID_str"] = patient_nodes["raw_key"].astype(str)

    visit_nodes = nodes[nodes["node_type"] == "visit"][["node_id", "raw_key"]].copy()
    visit_nodes["HADM_ID_str"] = visit_nodes["raw_key"].astype(str)

    # Patients present as patient nodes
    labeled_subj_ids = patient_nodes["SUBJECT_ID_str"].unique().tolist()
    print(f"Patients present as patient nodes: {len(labeled_subj_ids)}")

    # ------------------------------------------------------------------
    # Patient-based splits
    # ------------------------------------------------------------------
    train_subj, val_subj, test_subj = _split_patients(labeled_subj_ids)
    print(
        f"Patient splits: train={len(train_subj)}, val={len(val_subj)}, test={len(test_subj)}"
    )

    # SUBJECT_ID_str -> patient node_id
    subj_to_patient_nodeid: Dict[str, int] = dict(
        zip(
            patient_nodes["SUBJECT_ID_str"].tolist(),
            patient_nodes["node_id"].tolist(),
        )
    )

    # HADM_ID_str -> SUBJECT_ID_str from admissions
    hadm_to_subj: Dict[str, str] = dict(
        zip(admissions["HADM_ID_str"].tolist(), admissions["SUBJECT_ID_str"].tolist())
    )

    # SUBJECT_ID_str -> visit node_ids
    subj_to_visit_nodeids: Dict[str, List[int]] = {}
    for _, row in visit_nodes.iterrows():
        node_id = int(row["node_id"])
        hadm_str = row["HADM_ID_str"]
        subj_str = hadm_to_subj.get(hadm_str)
        if subj_str is None:
            continue
        subj_to_visit_nodeids.setdefault(subj_str, []).append(node_id)

    def _collect_nodes(subj_list: List[str]) -> List[int]:
        node_ids: List[int] = []
        for s in subj_list:
            # patient node
            pid = subj_to_patient_nodeid.get(s)
            if pid is not None:
                node_ids.append(pid)
            # visit nodes
            vids = subj_to_visit_nodeids.get(s, [])
            node_ids.extend(vids)
        return sorted(set(node_ids))

    train_nodes = _collect_nodes(train_subj)
    val_nodes = _collect_nodes(val_subj)
    test_nodes = _collect_nodes(test_subj)

    print(
        f"Node splits (patients + visits): "
        f"train={len(train_nodes)}, val={len(val_nodes)}, test={len(test_nodes)}"
    )

    # ------------------------------------------------------------------
    # Build label and mask tensors
    # ------------------------------------------------------------------
    num_nodes = nodes["node_id"].max() + 1
    num_tasks = 4  # [visit_mort, patient_mort, visit_readm, patient_readm]

    labels = torch.zeros((num_nodes, num_tasks), dtype=torch.float32)
    mask = torch.zeros((num_nodes, num_tasks), dtype=torch.float32)

    # Visit-level labels for visit nodes
    for _, row in visit_nodes.iterrows():
        node_id = int(row["node_id"])
        hadm_str = row["HADM_ID_str"]

        # visit mortality
        ym = hadm_to_visit_mort.get(hadm_str, None)
        if ym is not None:
            labels[node_id, 0] = float(ym)
            mask[node_id, 0] = 1.0

        # visit readmission
        yr = hadm_to_readm.get(hadm_str, None)
        if yr is not None:
            labels[node_id, 2] = float(yr)
            mask[node_id, 2] = 1.0

    # Patient-level labels for patient nodes
    for _, row in patient_nodes.iterrows():
        node_id = int(row["node_id"])
        subj_str = row["SUBJECT_ID_str"]

        # patient mortality
        ym = subj_to_patient_mort.get(subj_str, None)
        if ym is not None:
            labels[node_id, 1] = float(ym)
            mask[node_id, 1] = 1.0

        # patient readmission
        yr = subj_to_patient_readm.get(subj_str, None)
        if yr is not None:
            labels[node_id, 3] = float(yr)
            mask[node_id, 3] = 1.0

    print(f"Total labeled entries (visit_mort):   {int(mask[:, 0].sum().item())}")
    print(f"Total labeled entries (patient_mort): {int(mask[:, 1].sum().item())}")
    print(f"Total labeled entries (visit_readm):  {int(mask[:, 2].sum().item())}")
    print(f"Total labeled entries (patient_readm):{int(mask[:, 3].sum().item())}")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    ensure_dir(labels_out_path)
    ensure_dir(mask_out_path)
    torch.save(labels, labels_out_path)
    torch.save(mask, mask_out_path)
    print(f"Saved labels_outcomes -> {labels_out_path}")
    print(f"Saved labels_outcomes_mask -> {mask_out_path}")

    ensure_dir(train_ids_path)
    ensure_dir(val_ids_path)
    ensure_dir(test_ids_path)

    def _write_ids(path: str, ids: List[int]) -> None:
        with open(path, "w") as f:
            for nid in ids:
                f.write(f"{nid}\n")

    _write_ids(train_ids_path, train_nodes)
    _write_ids(val_ids_path, val_nodes)
    _write_ids(test_ids_path, test_nodes)

    print(f"Saved train_ids -> {train_ids_path}")
    print(f"Saved val_ids   -> {val_ids_path}")
    print(f"Saved test_ids  -> {test_ids_path}")

    return labels_out_path, mask_out_path, train_ids_path, val_ids_path, test_ids_path


if __name__ == "__main__":
    build_outcome_labels_and_splits()
