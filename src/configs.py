"""
configs.py

Helpers for loading project configuration from YAML files.

Configured for project root:
    /Users/amgzc/mujahid/projects/ehr-gnn/ehr-gnn

Config files:
    config/paths.yaml
    config/spark.yaml
    config/semantic_graph.yaml
    config/training.yaml
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _project_root() -> str:
    """
    Compute project root as the directory containing this file's parent 'src'.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, os.pardir))


def _load_yaml(relative_path: str) -> Dict[str, Any]:
    """
    Load a YAML file given a path relative to the project root.
    Raises a clear error if file is missing or malformed.
    """
    root = _project_root()
    full_path = os.path.join(root, relative_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Config file not found: {full_path}")

    with open(full_path, "r") as f:
        try:
            data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise RuntimeError(f"Failed to parse YAML config: {full_path}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {full_path} must be a mapping/dict.")
    return data


# ---------------------------------------------------------------------------
# Dataclasses for config sections
# ---------------------------------------------------------------------------

@dataclass
class PathsConfig:
    project_root: str

    data_raw_dir: str
    data_processed_dir: str
    data_graph_dir: str
    data_splits_dir: str

    admissions_csv: str
    diagnoses_csv: str
    procedures_csv: str
    prescriptions_csv: str
    labs_csv: str

    nodes_parquet: str
    ehr_edges_parquet: str
    contracted_edges_parquet: str
    full_graph_dgl: str

    metis_work_dir: str
    metis_graph_filename: str
    metis_num_parts: int
    metis_gpmetis_binary: str

    logs_root: str
    checkpoints_dir: str

    @staticmethod
    def from_yaml() -> "PathsConfig":
        cfg = _load_yaml(os.path.join("config", "paths.yaml"))

        project_root = cfg["project_root"]

        data_cfg = cfg.get("data", {})
        graph_files_cfg = cfg.get("graph_files", {})
        metis_cfg = cfg.get("metis", {})
        logs_cfg = cfg.get("logs", {})
        models_cfg = cfg.get("models", {})
        mimic_cfg = cfg.get("mimic", {})

        return PathsConfig(
            project_root=project_root,
            data_raw_dir=data_cfg["raw_dir"],
            data_processed_dir=data_cfg["processed_dir"],
            data_graph_dir=data_cfg["graph_dir"],
            data_splits_dir=data_cfg["splits_dir"],
            admissions_csv=mimic_cfg["admissions_csv"],
            diagnoses_csv=mimic_cfg["diagnoses_csv"],
            procedures_csv=mimic_cfg["procedures_csv"],
            prescriptions_csv=mimic_cfg["prescriptions_csv"],
            labs_csv=mimic_cfg["labs_csv"],
            nodes_parquet=graph_files_cfg["nodes_parquet"],
            ehr_edges_parquet=graph_files_cfg["ehr_edges_parquet"],
            contracted_edges_parquet=graph_files_cfg["contracted_edges_parquet"],
            full_graph_dgl=graph_files_cfg["full_graph_dgl"],
            metis_work_dir=metis_cfg["work_dir"],
            metis_graph_filename=metis_cfg["graph_filename"],
            metis_num_parts=int(metis_cfg["num_parts"]),
            metis_gpmetis_binary=metis_cfg["gpmetis_binary"],
            logs_root=logs_cfg["root"],
            checkpoints_dir=models_cfg["checkpoints_dir"],
        )

    # Convenience methods to build full paths
    def raw_csv_path(self, filename: str) -> str:
        return os.path.join(self.data_raw_dir, filename)

    def processed_path(self, filename: str) -> str:
        return os.path.join(self.data_processed_dir, filename)

    def graph_path(self, filename: str) -> str:
        return os.path.join(self.data_graph_dir, filename)

    def splits_path(self, filename: str) -> str:
        return os.path.join(self.data_splits_dir, filename)


@dataclass
class SparkConfig:
    app_name_prefix: str
    master: str
    config: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_yaml() -> "SparkConfig":
        cfg = _load_yaml(os.path.join("config", "spark.yaml"))

        return SparkConfig(
            app_name_prefix=cfg.get("app_name_prefix", "ehr-semantic-graph"),
            master=cfg.get("master", "local[*]"),
            config=cfg.get("config", {}) or {},
            env=cfg.get("env", {}) or {},
        )


@dataclass
class SemanticGraphConfig:
    diagnosis_digits: int
    procedure_digits: int

    use_mimic_flag: bool
    default_lab_bucket: str
    numeric_thresholds_enabled: bool
    numeric_thresholds_high_z: float
    numeric_thresholds_low_z: float

    edge_types: Dict[str, str]
    edge_weights: Dict[str, float]
    node_types: Dict[str, str]

    patient_contraction_enabled: bool
    patient_contraction_wmax: float

    label_level: str
    label_diagnosis_source: str
    label_min_labels_per_node: int

    @staticmethod
    def from_yaml() -> "SemanticGraphConfig":
        cfg = _load_yaml(os.path.join("config", "semantic_graph.yaml"))

        icd_cfg = cfg.get("icd_truncation", {})
        lab_cfg = cfg.get("lab_bucketing", {})
        num_thr = lab_cfg.get("numeric_thresholds", {})
        edge_types = cfg.get("edge_types", {})
        edge_weights = cfg.get("edge_weights", {})
        node_types = cfg.get("node_types", {})
        contraction_cfg = cfg.get("patient_contraction", {})
        labeling_cfg = cfg.get("labeling", {})

        return SemanticGraphConfig(
            diagnosis_digits=int(icd_cfg.get("diagnosis_digits", 3)),
            procedure_digits=int(icd_cfg.get("procedure_digits", 2)),
            use_mimic_flag=bool(lab_cfg.get("use_mimic_flag", True)),
            default_lab_bucket=str(lab_cfg.get("default_bucket", "normal")),
            numeric_thresholds_enabled=bool(num_thr.get("enabled", False)),
            numeric_thresholds_high_z=float(num_thr.get("high_zscore", 2.0)),
            numeric_thresholds_low_z=float(num_thr.get("low_zscore", -2.0)),
            edge_types=edge_types,
            edge_weights={k: float(v) for k, v in edge_weights.items()},
            node_types=node_types,
            patient_contraction_enabled=bool(contraction_cfg.get("enabled", True)),
            patient_contraction_wmax=float(contraction_cfg.get("wmax", 8.0)),
            label_level=str(labeling_cfg.get("level", "visit")),
            label_diagnosis_source=str(labeling_cfg.get("diagnosis_source", "DIAGNOSES_ICD")),
            label_min_labels_per_node=int(labeling_cfg.get("min_labels_per_node", 1)),
        )


@dataclass
class TrainingTaskConfig:
    name: str
    label_level: str
    target_node_type: str
    num_labels: int


@dataclass
class ModelConfig:
    type: str
    in_dim: int
    hidden_dim: int
    out_dim: int
    num_layers: int
    dropout: float
    aggregator: str


@dataclass
class OptimConfig:
    num_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    optimizer: str


@dataclass
class LossConfig:
    type: str
    pos_weight_path: Optional[str]


@dataclass
class SamplingConfig:
    use_neighbor_sampling: bool
    fanouts: List[int]
    num_workers: int


@dataclass
class EarlyStoppingConfig:
    enabled: bool
    patience: int
    metric: str
    mode: str


@dataclass
class PrecisionAtKConfig:
    ks: List[int]


@dataclass
class EvaluationConfig:
    metrics: List[str]
    precision_at_k: PrecisionAtKConfig
    eval_interval_epochs: int
    early_stopping: EarlyStoppingConfig


@dataclass
class TrainingConfig:
    task: TrainingTaskConfig
    model: ModelConfig
    training: OptimConfig
    loss: LossConfig
    sampling: SamplingConfig
    evaluation: EvaluationConfig

    @staticmethod
    def from_yaml() -> "TrainingConfig":
        cfg = _load_yaml(os.path.join("config", "training.yaml"))

        task_cfg = cfg.get("task", {})
        model_cfg = cfg.get("model", {})
        train_cfg = cfg.get("training", {})
        loss_cfg = cfg.get("loss", {})
        sampling_cfg = cfg.get("sampling", {})
        eval_cfg = cfg.get("evaluation", {})
        prec_cfg = eval_cfg.get("precision_at_k", {})
        es_cfg = eval_cfg.get("early_stopping", {})

        task = TrainingTaskConfig(
            name=str(task_cfg.get("name", "task")),
            label_level=str(task_cfg.get("label_level", "visit")),
            target_node_type=str(task_cfg.get("target_node_type", "visit")),
            num_labels=int(task_cfg.get("num_labels", 1)),
        )

        model = ModelConfig(
            type=str(model_cfg.get("type", "graphsage")),
            in_dim=int(model_cfg.get("in_dim", 64)),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            out_dim=int(model_cfg.get("out_dim", task.num_labels)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.5)),
            aggregator=str(model_cfg.get("aggregator", "mean")),
        )

        training = OptimConfig(
            num_epochs=int(train_cfg.get("num_epochs", 10)),
            batch_size=int(train_cfg.get("batch_size", 1024)),
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 5e-4)),
            optimizer=str(train_cfg.get("optimizer", "adam")),
        )

        loss = LossConfig(
            type=str(loss_cfg.get("type", "binary_cross_entropy_with_logits")),
            pos_weight_path=loss_cfg.get("pos_weight_path", None),
        )

        sampling = SamplingConfig(
            use_neighbor_sampling=bool(sampling_cfg.get("use_neighbor_sampling", True)),
            fanouts=list(sampling_cfg.get("fanouts", [10, 10])),
            num_workers=int(sampling_cfg.get("num_workers", 4)),
        )

        precision_at_k = PrecisionAtKConfig(
            ks=list(prec_cfg.get("ks", [5, 10]))
        )

        early_stopping = EarlyStoppingConfig(
            enabled=bool(es_cfg.get("enabled", True)),
            patience=int(es_cfg.get("patience", 5)),
            metric=str(es_cfg.get("metric", "precision_at_k@5")),
            mode=str(es_cfg.get("mode", "max")),
        )

        evaluation = EvaluationConfig(
            metrics=list(eval_cfg.get("metrics", ["precision_at_k"])),
            precision_at_k=precision_at_k,
            eval_interval_epochs=int(eval_cfg.get("eval_interval_epochs", 1)),
            early_stopping=early_stopping,
        )

        return TrainingConfig(
            task=task,
            model=model,
            training=training,
            loss=loss,
            sampling=sampling,
            evaluation=evaluation,
        )


# ---------------------------------------------------------------------------
# Convenience top-level loader functions
# ---------------------------------------------------------------------------

def load_paths_config() -> PathsConfig:
    return PathsConfig.from_yaml()


def load_spark_config() -> SparkConfig:
    return SparkConfig.from_yaml()


def load_semantic_graph_config() -> SemanticGraphConfig:
    return SemanticGraphConfig.from_yaml()


def load_training_config() -> TrainingConfig:
    return TrainingConfig.from_yaml()
