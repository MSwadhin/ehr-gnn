import os

from pyspark.sql import functions as F, types as T, DataFrame

from src.configs import load_paths_config, load_semantic_graph_config
from src.utils.spark_utils import create_spark_session


def patient_contraction() -> str:
    """
    Perform patient-level graph contraction.

    Input:
      data/graph/ehr_edges.parquet
        columns: src_id, dst_id, src_type, dst_type, edge_type, weight

    Uses:
      semantic_graph.yaml:
        - edge_types.patient_visit
        - edge_types.visit_diagnosis
        - edge_types.visit_procedure
        - edge_types.visit_medication
        - edge_types.visit_lab
        - patient_contraction.wmax

    Output:
      data/graph/contracted_edges.parquet
        columns: src_id, dst_id, src_type, dst_type, edge_type, weight

      where src_type == 'patient' and dst_type in {diagnosis, procedure, medication, lab_bucket}
    """
    paths = load_paths_config()
    sem_cfg = load_semantic_graph_config()

    spark = create_spark_session("patient-contraction")

    graph_dir = paths.data_graph_dir
    edges_path = os.path.join(graph_dir, paths.ehr_edges_parquet)
    contracted_out_path = os.path.join(graph_dir, paths.contracted_edges_parquet)

    print(f"Loading graph edges from {edges_path}")
    edges = spark.read.parquet(edges_path)

    expected_cols = {"src_id", "dst_id", "src_type", "dst_type", "edge_type", "weight"}
    missing = expected_cols - set(edges.columns)
    if missing:
        raise ValueError(f"Graph edges file missing columns: {missing}")

    # ----------------------------------------------------------------------
    # 1) Identify edge_type strings from config
    # ----------------------------------------------------------------------
    etypes = sem_cfg.edge_types
    et_patient_visit = etypes["patient_visit"]
    et_visit_diag = etypes["visit_diagnosis"]
    et_visit_proc = etypes["visit_procedure"]
    et_visit_med = etypes["visit_medication"]
    et_visit_lab = etypes["visit_lab"]

    wmax = sem_cfg.patient_contraction_wmax

    print(f"Using wmax = {wmax} for patient-contraction")

    # ----------------------------------------------------------------------
    # 2) Extract patient-visit mapping
    #    Expect patient-visit edges: src_type='patient', dst_type='visit'
    # ----------------------------------------------------------------------
    pv_edges = (
        edges
        .filter(F.col("edge_type") == F.lit(et_patient_visit))
        .select(
            F.col("src_id").cast(T.LongType()).alias("patient_id"),
            F.col("dst_id").cast(T.LongType()).alias("visit_id"),
        )
        .dropDuplicates()
    )

    num_patients = pv_edges.select("patient_id").distinct().count()
    num_visits = pv_edges.select("visit_id").distinct().count()
    print(f"Patient-visit mapping: {num_patients} patients, {num_visits} visits")

    # ----------------------------------------------------------------------
    # 3) Collect visit -> clinical edges (diagnosis/procedure/medication/lab)
    # ----------------------------------------------------------------------
    visit_edge_types = [et_visit_diag, et_visit_proc, et_visit_med, et_visit_lab]

    visit_clinical_edges = (
        edges
        .filter(F.col("edge_type").isin(visit_edge_types))
        .filter(F.col("src_type") == F.lit("visit"))
        .select(
            F.col("src_id").cast(T.LongType()).alias("visit_id"),
            F.col("dst_id").cast(T.LongType()).alias("clinical_id"),
            "edge_type",
            "dst_type",
            F.col("weight").cast(T.DoubleType()).alias("weight"),
        )
    )

    num_visit_clinical = visit_clinical_edges.count()
    print(f"Visit→clinical edges before contraction: {num_visit_clinical}")

    # ----------------------------------------------------------------------
    # 4) Join visit→clinical with patient-visit to get patient→clinical candidates
    # ----------------------------------------------------------------------
    pv_joined = (
        visit_clinical_edges
        .join(pv_edges, on="visit_id", how="inner")
        .select(
            "patient_id",
            "clinical_id",
            "dst_type",
            "edge_type",
            "weight",
        )
    )

    num_pv_joined = pv_joined.count()
    print(f"Visit→clinical edges with patient mapping: {num_pv_joined}")

    # ----------------------------------------------------------------------
    # 5) Aggregate edges per (patient, clinical, edge_type) and apply wmax cap
    # ----------------------------------------------------------------------
    agg = (
        pv_joined
        .groupBy("patient_id", "clinical_id", "dst_type", "edge_type")
        .agg(F.sum("weight").alias("weight_sum"))
        .withColumn(
            "weight_capped",
            F.when(F.col("weight_sum") > F.lit(float(wmax)), F.lit(float(wmax)))
             .otherwise(F.col("weight_sum")),
        )
    )

    contracted_edges = (
        agg
        .select(
            F.col("patient_id").alias("src_id"),
            F.col("clinical_id").alias("dst_id"),
            F.lit("patient").alias("src_type"),
            F.col("dst_type").alias("dst_type"),
            F.col("edge_type").alias("edge_type"),
            F.col("weight_capped").alias("weight"),
        )
    )

    num_contracted = contracted_edges.count()
    print(f"Contracted patient→clinical edges: {num_contracted}")

    # ----------------------------------------------------------------------
    # 6) Write contracted edges
    # ----------------------------------------------------------------------
    print(f"Writing contracted edges -> {contracted_out_path}")
    contracted_edges.write.mode("overwrite").parquet(contracted_out_path)

    spark.stop()
    return contracted_out_path


if __name__ == "__main__":
    patient_contraction()
