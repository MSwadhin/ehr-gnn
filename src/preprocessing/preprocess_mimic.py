import os
from typing import Tuple

import pandas as pd

from src.configs import load_paths_config, load_semantic_graph_config
from src.utils.io_utils import read_csv, write_parquet



def _truncate_icd_code(code: str, num_digits: int) -> str:
    """
    Truncate an ICD code string to the given number of digits.
    MIMIC ICD-9 codes are often stored as strings like '4280', '25000', etc.
    We simply take the first num_digits characters of the alphanumeric code.
    """
    if not isinstance(code, str):
        return ""
    code = code.strip()
    if not code:
        return ""
    # Remove dots if present (e.g., '428.0' -> '4280')
    code = code.replace(".", "")
    return code[:num_digits]


def preprocess_diagnoses(diag_df: pd.DataFrame, num_digits: int) -> pd.DataFrame:
    """
    Preprocess DIAGNOSES_ICD:
      - keep SUBJECT_ID, HADM_ID, ICD9_CODE
      - truncate ICD code
      - drop rows with empty truncated codes
    """
    required_cols = {"SUBJECT_ID", "HADM_ID", "ICD9_CODE"}
    missing = required_cols - set(diag_df.columns)
    if missing:
        raise ValueError(f"DIAGNOSES_ICD missing columns: {missing}")

    df = diag_df.copy()
    df["ICD9_TRUNC"] = df["ICD9_CODE"].astype(str).apply(
        lambda x: _truncate_icd_code(x, num_digits)
    )
    df = df[df["ICD9_TRUNC"] != ""]
    return df[["SUBJECT_ID", "HADM_ID", "ICD9_CODE", "ICD9_TRUNC"]]


def preprocess_procedures(proc_df: pd.DataFrame, num_digits: int) -> pd.DataFrame:
    """
    Preprocess PROCEDURES_ICD similarly to diagnoses.
    """
    required_cols = {"SUBJECT_ID", "HADM_ID", "ICD9_CODE"}
    missing = required_cols - set(proc_df.columns)
    if missing:
        raise ValueError(f"PROCEDURES_ICD missing columns: {missing}")

    df = proc_df.copy()
    df["ICD9_TRUNC"] = df["ICD9_CODE"].astype(str).apply(
        lambda x: _truncate_icd_code(x, num_digits)
    )
    df = df[df["ICD9_TRUNC"] != ""]
    return df[["SUBJECT_ID", "HADM_ID", "ICD9_CODE", "ICD9_TRUNC"]]


def preprocess_admissions(adm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic ADMISSIONS cleanup:
      - keep SUBJECT_ID, HADM_ID, ADMITTIME, DISCHTIME
      - optionally filter to first admission per HADM_ID (already unique)
    """
    required_cols = {"SUBJECT_ID", "HADM_ID", "ADMITTIME"}
    missing = required_cols - set(adm_df.columns)
    if missing:
        raise ValueError(f"ADMISSIONS missing columns: {missing}")

    df = adm_df.copy()
    # Convert timestamps to datetime if not already
    for col in ["ADMITTIME", "DISCHTIME"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # HADM_ID is already unique in ADMISSIONS
    return df[["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"]]


def normalize_med_name(name: str) -> str:
    """
    Very simple medication name normalization:
      - lowercase
      - strip whitespace
    You can later extend this with regex to strip dose, etc.
    """
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def preprocess_prescriptions(rx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess PRESCRIPTIONS:
      - keep SUBJECT_ID, HADM_ID, DRUG
      - normalize DRUG name
    """
    required_cols = {"SUBJECT_ID", "HADM_ID", "DRUG"}
    missing = required_cols - set(rx_df.columns)
    if missing:
        raise ValueError(f"PRESCRIPTIONS missing columns: {missing}")

    df = rx_df.copy()
    df["DRUG_NORM"] = df["DRUG"].apply(normalize_med_name)
    df = df[df["DRUG_NORM"] != ""]
    return df[["SUBJECT_ID", "HADM_ID", "DRUG", "DRUG_NORM"]]


def preprocess_labs(lab_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess LABEVENTS:
      - keep SUBJECT_ID, HADM_ID, ITEMID, FLAG if available
      - we will bucket into lab buckets later (using FLAG or thresholds)
    """
    required_cols = {"SUBJECT_ID", "HADM_ID", "ITEMID"}
    missing = required_cols - set(lab_df.columns)
    if missing:
        raise ValueError(f"LABEVENTS missing columns: {missing}")

    cols = ["SUBJECT_ID", "HADM_ID", "ITEMID"]
    if "FLAG" in lab_df.columns:
        cols.append("FLAG")
    df = lab_df[cols].copy()
    return df


def run_preprocessing() -> Tuple[str, str, str, str, str]:
    """
    Main entrypoint for preprocessing:
      - reads raw CSVs from data/raw
      - writes cleaned parquet files to data/processed
    Returns tuple of written paths.
    """
    paths = load_paths_config()
    sem_cfg = load_semantic_graph_config()

    # Raw CSV locations
    raw_dir = paths.data_raw_dir

    admissions_csv = os.path.join(raw_dir, paths.admissions_csv)
    diagnoses_csv = os.path.join(raw_dir, paths.diagnoses_csv)
    procedures_csv = os.path.join(raw_dir, paths.procedures_csv)
    prescriptions_csv = os.path.join(raw_dir, paths.prescriptions_csv)
    labs_csv = os.path.join(raw_dir, paths.labs_csv)

    print(f"Reading ADMISSIONS from {admissions_csv}")
    adm_df = read_csv(admissions_csv)

    print(f"Reading DIAGNOSES from {diagnoses_csv}")
    diag_df = read_csv(diagnoses_csv)

    print(f"Reading PROCEDURES from {procedures_csv}")
    proc_df = read_csv(procedures_csv)

    print(f"Reading PRESCRIPTIONS from {prescriptions_csv}")
    rx_df = read_csv(prescriptions_csv)

    print(f"Reading LABEVENTS from {labs_csv}")
    lab_df = read_csv(labs_csv)

    # ICD truncation digits from config
    diag_digits = sem_cfg.diagnosis_digits
    proc_digits = sem_cfg.procedure_digits

    print(f"Preprocessing ADMISSIONS...")
    adm_clean = preprocess_admissions(adm_df)

    print(f"Preprocessing DIAGNOSES (truncate to {diag_digits} digits)...")
    diag_clean = preprocess_diagnoses(diag_df, diag_digits)

    print(f"Preprocessing PROCEDURES (truncate to {proc_digits} digits)...")
    proc_clean = preprocess_procedures(proc_df, proc_digits)

    print(f"Preprocessing PRESCRIPTIONS...")
    rx_clean = preprocess_prescriptions(rx_df)

    print(f"Preprocessing LABEVENTS...")
    lab_clean = preprocess_labs(lab_df)

    # Output paths
    proc_dir = paths.data_processed_dir
    adm_out = os.path.join(proc_dir, "admissions.parquet")
    diag_out = os.path.join(proc_dir, "diagnoses.parquet")
    proc_out = os.path.join(proc_dir, "procedures.parquet")
    rx_out = os.path.join(proc_dir, "prescriptions.parquet")
    lab_out = os.path.join(proc_dir, "labs.parquet")

    print(f"Writing ADMISSIONS -> {adm_out}")
    write_parquet(adm_clean, adm_out)

    print(f"Writing DIAGNOSES -> {diag_out}")
    write_parquet(diag_clean, diag_out)

    print(f"Writing PROCEDURES -> {proc_out}")
    write_parquet(proc_clean, proc_out)

    print(f"Writing PRESCRIPTIONS -> {rx_out}")
    write_parquet(rx_clean, rx_out)

    print(f"Writing LABEVENTS -> {lab_out}")
    write_parquet(lab_clean, lab_out)

    return adm_out, diag_out, proc_out, rx_out, lab_out


if __name__ == "__main__":
    run_preprocessing()
