"""
Severity Prediction Project
===========================
End-to-end Machine Learning pipeline for severity prediction.
Covers EDA, Preprocessing, Feature Engineering, Model Training, Evaluation, and Inference.
"""

from .model_utils import (
    get_project_root,
    load_model_artifacts,
    save_model_artifacts,
    detect_target_column,
)
from .feature_engineering import engineer_features
from .preprocessing import DataPreprocessor, clean_dataset
from .predict import SeverityPredictor, predict_severity

__all__ = [
    "get_project_root",
    "load_model_artifacts",
    "save_model_artifacts",
    "detect_target_column",
    "engineer_features",
    "DataPreprocessor",
    "clean_dataset",
    "SeverityPredictor",
    "predict_severity",
]
