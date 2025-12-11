import os

from pyspark.sql import DataFrame, functions as F, types as T

from src.configs import load_paths_config, load_semantic_graph_config
from src.utils.spark_utils import create_spark_session



def _bucket_lab_flag(use_flag: bool):
    """
    Return a Spark UDF that maps FLAG to a lab bucket string.
    Very simple: if FLAG present, lowercase; else "normal".
    Later you can replace with a more nuanced mapping.
    """
    def _map(flag: str) -> str:
        if not use_flag:
            return "normal"
        if flag is None:
            return "normal"
        flag = str(flag).strip().lower()
        # Many MIMIC flags are 'abnormal', 'high', 'low'
        if flag in ("abnormal", "high", "low"):
            return flag
        return "normal"

    return F.udf(_map, T.StringType())


def build_semantic_edges() -> str:
    """
    Build heterogeneous EHR graph edges with semantic weights using Spark.

    Output:
      data/processed/ehr_edges.parquet with columns:
        src_type, src_key, dst_type, dst_key, edge_type, weight
    """
    paths = load_paths_config()
    sem_cfg = load_semantic_graph_config()

    spark = create_spark_session("build-semantic-edges")

    processed_dir = paths.data_processed_dir

    adm_path = os.path.join(processed_dir, "admissions.parquet")
    diag_path = os.path.join(processed_dir, "diagnoses.parquet")
    proc_path = os.path.join(processed_dir, "procedures.parquet")
    rx_path   = os.path.join(processed_dir, "prescriptions.parquet")
    lab_path  = os.path.join(processed_dir, "labs.parquet")

    print(f"Loading processed tables from {processed_dir}")
    admissions = spark.read.parquet(adm_path)
    diagnoses  = spark.read.parquet(diag_path)
    procedures = spark.read.parquet(proc_path)
    prescriptions = spark.read.parquet(rx_path)
    labs = spark.read.parquet(lab_path)

    # ----------------------------------------------------------------------
    # 1) Define visit-id and patient-id keys
    # For now, treat:
    #   patient_key = SUBJECT_ID
    #   visit_key   = HADM_ID
    # ----------------------------------------------------------------------
    # patient_visit edges
    edge_type_pv = sem_cfg.edge_types["patient_visit"]
    w_pv = sem_cfg.edge_weights[edge_type_pv]

    patient_visit_edges = (
        admissions
        .select(
            F.col("SUBJECT_ID").cast(T.StringType()).alias("patient_key"),
            F.col("HADM_ID").cast(T.StringType()).alias("visit_key"),
        )
        .dropna()
        .dropDuplicates()
        .withColumn("src_type", F.lit("patient"))
        .withColumn("src_key", F.col("patient_key"))
        .withColumn("dst_type", F.lit("visit"))
        .withColumn("dst_key", F.col("visit_key"))
        .withColumn("edge_type", F.lit(edge_type_pv))
        .withColumn("weight", F.lit(float(w_pv)))
        .select("src_type", "src_key", "dst_type", "dst_key", "edge_type", "weight")
    )

    # ----------------------------------------------------------------------
    # 2) visit - diagnosis edges
    # ----------------------------------------------------------------------
    edge_type_vd = sem_cfg.edge_types["visit_diagnosis"]
    w_vd = sem_cfg.edge_weights[edge_type_vd]

    visit_diag_edges = (
        diagnoses
        .select(
            F.col("HADM_ID").cast(T.StringType()).alias("visit_key"),
            F.col("ICD9_TRUNC").cast(T.StringType()).alias("diag_code"),
        )
        .dropna()
        .dropDuplicates()
        .withColumn("src_type", F.lit("visit"))
        .withColumn("src_key", F.col("visit_key"))
        .withColumn("dst_type", F.lit("diagnosis"))
        .withColumn("dst_key", F.col("diag_code"))
        .withColumn("edge_type", F.lit(edge_type_vd))
        .withColumn("weight", F.lit(float(w_vd)))
        .select("src_type", "src_key", "dst_type", "dst_key", "edge_type", "weight")
    )

    # ----------------------------------------------------------------------
    # 3) visit - procedure edges
    # ----------------------------------------------------------------------
    edge_type_vp = sem_cfg.edge_types["visit_procedure"]
    w_vp = sem_cfg.edge_weights[edge_type_vp]

    visit_proc_edges = (
        procedures
        .select(
            F.col("HADM_ID").cast(T.StringType()).alias("visit_key"),
            F.col("ICD9_TRUNC").cast(T.StringType()).alias("proc_code"),
        )
        .dropna()
        .dropDuplicates()
        .withColumn("src_type", F.lit("visit"))
        .withColumn("src_key", F.col("visit_key"))
        .withColumn("dst_type", F.lit("procedure"))
        .withColumn("dst_key", F.col("proc_code"))
        .withColumn("edge_type", F.lit(edge_type_vp))
        .withColumn("weight", F.lit(float(w_vp)))
        .select("src_type", "src_key", "dst_type", "dst_key", "edge_type", "weight")
    )

    # ----------------------------------------------------------------------
    # 4) visit - medication edges
    # ----------------------------------------------------------------------
    edge_type_vm = sem_cfg.edge_types["visit_medication"]
    w_vm = sem_cfg.edge_weights[edge_type_vm]

    visit_med_edges = (
        prescriptions
        .select(
            F.col("HADM_ID").cast(T.StringType()).alias("visit_key"),
            F.col("DRUG_NORM").cast(T.StringType()).alias("drug_norm"),
        )
        .dropna()
        .dropDuplicates()
        .withColumn("src_type", F.lit("visit"))
        .withColumn("src_key", F.col("visit_key"))
        .withColumn("dst_type", F.lit("medication"))
        .withColumn("dst_key", F.col("drug_norm"))
        .withColumn("edge_type", F.lit(edge_type_vm))
        .withColumn("weight", F.lit(float(w_vm)))
        .select("src_type", "src_key", "dst_type", "dst_key", "edge_type", "weight")
    )

    # ----------------------------------------------------------------------
    # 5) visit - lab_bucket edges
    # ----------------------------------------------------------------------
    edge_type_vl = sem_cfg.edge_types["visit_lab"]
    w_vl = sem_cfg.edge_weights[edge_type_vl]
    use_flag = sem_cfg.use_mimic_flag

    bucket_udf = _bucket_lab_flag(use_flag)

    labs_with_bucket = (
        labs
        .select(
            F.col("HADM_ID").cast(T.StringType()).alias("visit_key"),
            F.col("ITEMID").cast(T.StringType()).alias("itemid"),
            F.col("FLAG").cast(T.StringType()).alias("FLAG") if "FLAG" in labs.columns else F.lit(None).alias("FLAG")
        )
        .withColumn("lab_bucket", bucket_udf(F.col("FLAG")))
        .dropna(subset=["visit_key", "lab_bucket"])
        .dropDuplicates(["visit_key", "itemid", "lab_bucket"])
    )

    visit_lab_edges = (
        labs_with_bucket
        .withColumn("src_type", F.lit("visit"))
        .withColumn("src_key", F.col("visit_key"))
        .withColumn("dst_type", F.lit("lab_bucket"))
        .withColumn("dst_key", F.col("lab_bucket"))
        .withColumn("edge_type", F.lit(edge_type_vl))
        .withColumn("weight", F.lit(float(w_vl)))
        .select("src_type", "src_key", "dst_type", "dst_key", "edge_type", "weight")
    )

    # ----------------------------------------------------------------------
    # Union all edge types into a single edge DataFrame
    # ----------------------------------------------------------------------
    edges: DataFrame = (
        patient_visit_edges
        .unionByName(visit_diag_edges)
        .unionByName(visit_proc_edges)
        .unionByName(visit_med_edges)
        .unionByName(visit_lab_edges)
    )

    # Optionally, you could aggregate to ensure a unique edge per (src,dst,edge_type)
    edges = (
        edges
        .groupBy("src_type", "src_key", "dst_type", "dst_key", "edge_type")
        .agg(F.sum("weight").alias("weight"))
    )

    out_path = os.path.join(paths.data_processed_dir, "ehr_edges.parquet")
    print(f"Writing semantic edges -> {out_path}")
    edges.write.mode("overwrite").parquet(out_path)

    spark.stop()
    return out_path


if __name__ == "__main__":
    build_semantic_edges()
