from dataclasses import dataclass, field
from pathlib import Path


def validate_data_source(value):
    if value not in {"DAAS", "Sheets"}:
        raise ValueError("data_source must be either 'DAAS' or 'Sheets'")


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_source: str = field(metadata={"validator": validate_data_source})
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

    def __post_init__(self):
        validate_data_source(self.data_source)


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    test_size: float
    train_size: float


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    alpha: float
    l1_ratio: float
    target_column: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str