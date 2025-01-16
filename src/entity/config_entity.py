from dataclasses import dataclass
from pathlib import Path

@dataclass
class dataIngestionConfig:
    root_dir: Path
    source_url: str
    local_save : Path
    train_data_path : Path
    test_data_path : Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    preprocessor_path: Path
    transformed_train_path: Path
    transformed_test_path: Path
    train_target_path: Path
    test_target_path: Path

@dataclass
class DataValidationConfig:
    root_dir:Path
    STATUS_FILE:str
    data_dir:Path
    all_schema:dict

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_transformed_path: Path
    test_target_path: Path
    best_model_path: Path
    evaluation_results_path: Path
    mlflow_uri: str


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    train_target_path: Path
    test_target_path: Path
    model_dir: Path
    model_params: dict
    cv_folds: int