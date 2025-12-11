import os
import pandas as pd

from src.configs import load_paths_config
from src.utils.io_utils import ensure_dir


def rebuild_admissions():
    paths = load_paths_config()

    # TODO: set this to your actual raw ADMISSIONS CSV path
    # e.g. "/Users/amgzc/mimic/ADMISSIONS.csv" or wherever you keep it
    RAW_ADM_PATH = "/Users/amgzc/mujahid/projects/ehr-gnn/ehr-gnn/data/raw/ADMISSIONS.csv"

    out_path = os.path.join(paths.data_processed_dir, "admissions.parquet")

    print(f"Reading raw admissions from {RAW_ADM_PATH}")
    adm = pd.read_csv(RAW_ADM_PATH)

    print("Raw admissions columns:")
    print(adm.columns.tolist())

    # Keep the columns we actually need downstream.
    # If DEATHTIME exists, we keep it; otherwise we ignore it.
    required = ["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "HOSPITAL_EXPIRE_FLAG"]
    optional = ["DEATHTIME"]

    cols_to_keep = []
    for c in required + optional:
        if c in adm.columns:
            cols_to_keep.append(c)

    missing_required = [c for c in required if c not in adm.columns]
    if missing_required:
        raise ValueError(f"Raw ADMISSIONS.csv is missing required columns: {missing_required}")

    adm = adm[cols_to_keep].copy()

    # Parse datetimes
    adm["ADMITTIME"] = pd.to_datetime(adm["ADMITTIME"])
    adm["DISCHTIME"] = pd.to_datetime(adm["DISCHTIME"])
    if "DEATHTIME" in adm.columns:
        adm["DEATHTIME"] = pd.to_datetime(adm["DEATHTIME"])

    # Make sure mortality flag is int 0/1
    adm["HOSPITAL_EXPIRE_FLAG"] = adm["HOSPITAL_EXPIRE_FLAG"].fillna(0).astype(int)

    ensure_dir(out_path)
    adm.to_parquet(out_path, index=False)
    print(f"Wrote processed admissions -> {out_path}")


if __name__ == "__main__":
    rebuild_admissions()
