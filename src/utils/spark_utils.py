import os
from typing import Optional

from pyspark.sql import SparkSession
from src.configs import load_spark_config


def create_spark_session(app_name_suffix: Optional[str] = None) -> SparkSession:
    """
    Create a SparkSession based on config/spark.yaml.

    app_name_suffix: optional string appended to the configured app_name_prefix.
    """
    spark_cfg = load_spark_config()

    app_name = spark_cfg.app_name_prefix
    if app_name_suffix:
        app_name = f"{app_name}-{app_name_suffix}"

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(spark_cfg.master)
    )

    # Apply Spark config key/values
    for key, value in spark_cfg.config.items():
        builder = builder.config(key, value)

    # Apply env vars for the Spark driver (Java/Python paths)
    for key, value in spark_cfg.env.items():
        if value is not None:
            os.environ[key] = value

    spark = builder.getOrCreate()
    return spark
