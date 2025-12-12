import os
import numpy as np
import pandas as pd

from src.configs import load_paths_config


def _read_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def build_patient_visit_features():
    paths = load_paths_config()
    # FIX: use processed_path
    proc_dir = "data/processed"

    # ---- Load core tables ----
    admissions = _read_parquet(os.path.join(proc_dir, "admissions.parquet"))
    patients = _read_parquet(os.path.join(proc_dir, "patients.parquet"))
    diagnoses = _read_parquet(os.path.join(proc_dir, "diagnoses_icd.parquet"))
    procedures = _read_parquet(os.path.join(proc_dir, "procedures_icd.parquet"))
    prescriptions = _read_parquet(os.path.join(proc_dir, "prescriptions.parquet"))

    # Normalize column names (upper)
    admissions.columns = [c.upper() for c in admissions.columns]
    patients.columns = [c.upper() for c in patients.columns]
    diagnoses.columns = [c.upper() for c in diagnoses.columns]
    procedures.columns = [c.upper() for c in procedures.columns]
    prescriptions.columns = [c.upper() for c in prescriptions.columns]

    for col in ["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"]:
        if col not in admissions.columns:
            raise ValueError(f"admissions.parquet missing column {col}")

    # Cast times
    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"])
    admissions["DISCHTIME"] = pd.to_datetime(admissions["DISCHTIME"])
    if "DOB" in patients.columns:
        patients["DOB"] = pd.to_datetime(patients["DOB"])

    # -----------------------
    # Visit-level features
    # -----------------------
    adm = admissions.copy()

    # Age at admission
    if "DOB" in patients.columns:
        adm = adm.merge(
            patients[["SUBJECT_ID", "DOB"]],
            on="SUBJECT_ID",
            how="left",
        )
        adm["AGE_AT_ADMIT"] = (adm["ADMITTIME"] - adm["DOB"]).dt.days / 365.25
        adm["AGE_AT_ADMIT"] = adm["AGE_AT_ADMIT"].clip(lower=0, upper=120).fillna(0)
    else:
        adm["AGE_AT_ADMIT"] = 0.0

    # Length of stay (days)
    adm["LOS_DAYS"] = (adm["DISCHTIME"] - adm["ADMITTIME"]).dt.total_seconds() / 86400.0
    adm["LOS_DAYS"] = adm["LOS_DAYS"].clip(lower=0).fillna(0)

    # Counts from diagnoses / procedures / prescriptions
    for df, name in [
        (diagnoses, "N_DIAGNOSES"),
        (procedures, "N_PROCS"),
        (prescriptions, "N_MEDS"),
    ]:
        if "HADM_ID" in df.columns:
            cnt = df.groupby("HADM_ID").size().rename(name).reset_index()
            adm = adm.merge(cnt, on="HADM_ID", how="left")
            adm[name] = adm[name].fillna(0).astype(np.float32)
        else:
            adm[name] = 0.0

    # Prior visits and time since last visit
    adm = adm.sort_values(["SUBJECT_ID", "ADMITTIME"])
    adm["N_PRIOR_VISITS"] = (
        adm.groupby("SUBJECT_ID").cumcount().astype(np.float32)
    )

    adm["PREV_DISCHTIME"] = adm.groupby("SUBJECT_ID")["DISCHTIME"].shift(1)
    delta = (adm["ADMITTIME"] - adm["PREV_DISCHTIME"]).dt.total_seconds() / 86400.0
    adm["DAYS_SINCE_LAST_VISIT"] = delta.fillna(-1.0)

    # Categorical one-hot
    cat_cols = []
    for c in ["ADMISSION_TYPE", "ADMISSION_LOCATION", "DISCHARGE_LOCATION"]:
        if c in adm.columns:
            cat_cols.append(c)

    if cat_cols:
        dummies = pd.get_dummies(adm[cat_cols], prefix=[c.lower() for c in cat_cols])
        adm = pd.concat([adm, dummies], axis=1)

    base_cols = [
        "SUBJECT_ID",
        "HADM_ID",
        "AGE_AT_ADMIT",
        "LOS_DAYS",
        "N_DIAGNOSES",
        "N_PROCS",
        "N_MEDS",
        "N_PRIOR_VISITS",
        "DAYS_SINCE_LAST_VISIT",
    ]
    dummy_cols = [
        c
        for c in adm.columns
        if any(c.startswith(p) for p in ["admission_type_", "admission_location_", "discharge_location_"])
    ]

    visit_feat_cols = base_cols + dummy_cols
    visit_features = adm[visit_feat_cols].copy()

    visit_out = os.path.join(proc_dir, "visit_features.parquet")
    visit_features.to_parquet(visit_out, index=False)
    print(f"Saved visit_features to {visit_out}, shape={visit_features.shape}")

    # -----------------------
    # Patient-level features
    # -----------------------
    pat = patients.copy()

    visits_per_pat = admissions.groupby("SUBJECT_ID")["HADM_ID"].nunique().rename("TOTAL_VISITS")
    pat = pat.merge(visits_per_pat, on="SUBJECT_ID", how="left")
    pat["TOTAL_VISITS"] = pat["TOTAL_VISITS"].fillna(0).astype(np.float32)

    last_adm = admissions.groupby("SUBJECT_ID")["ADMITTIME"].max().rename("LAST_ADMITTIME").reset_index()
    pat = pat.merge(last_adm, on="SUBJECT_ID", how="left")
    if "DOB" in pat.columns:
        pat["DOB"] = pd.to_datetime(pat["DOB"])
        pat["AGE_AT_LAST_ADMIT"] = (pat["LAST_ADMITTIME"] - pat["DOB"]).dt.days / 365.25
        pat["AGE_AT_LAST_ADMIT"] = pat["AGE_AT_LAST_ADMIT"].clip(lower=0, upper=120).fillna(0)
    else:
        pat["AGE_AT_LAST_ADMIT"] = 0.0

    if "SUBJECT_ID" in diagnoses.columns:
        diag_per_pat = diagnoses.groupby("SUBJECT_ID").size().rename("TOTAL_DIAGNOSES")
        pat = pat.merge(diag_per_pat, on="SUBJECT_ID", how="left")
        pat["TOTAL_DIAGNOSES"] = pat["TOTAL_DIAGNOSES"].fillna(0).astype(np.float32)
    else:
        pat["TOTAL_DIAGNOSES"] = 0.0

    cat_cols = []
    for c in ["GENDER", "ETHNICITY"]:
        if c in pat.columns:
            cat_cols.append(c)

    if cat_cols:
        dummies = pd.get_dummies(pat[cat_cols], prefix=[c.lower() for c in cat_cols])
        pat = pd.concat([pat, dummies], axis=1)

    patient_base = ["SUBJECT_ID", "AGE_AT_LAST_ADMIT", "TOTAL_VISITS", "TOTAL_DIAGNOSES"]
    patient_dummy = [c for c in pat.columns if c.startswith("gender_") or c.startswith("ethnicity_")]

    patient_feat_cols = patient_base + patient_dummy
    patient_features = pat[patient_feat_cols].copy()

    patient_out = os.path.join(proc_dir, "patient_features.parquet")
    patient_features.to_parquet(patient_out, index=False)
    print(f"Saved patient_features to {patient_out}, shape={patient_features.shape}")


if __name__ == "__main__":
    build_patient_visit_features()
