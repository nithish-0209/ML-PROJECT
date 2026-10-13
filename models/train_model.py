import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from .feature_engineering import engineer_features
from .model_utils import (
    detect_target_column,
    get_artifacts_dir,
    get_data_path,
    save_model_artifacts,
)
from .preprocessing import DataPreprocessor, clean_dataset, prepare_train_test_split

logger = logging.getLogger("SeverityPrediction.Training")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def train_and_evaluate_models(
    data_path: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    target_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train and rigorously evaluate multiple supervised learning models.
    Handles class imbalance, prevents data leakage, computes comprehensive metrics,
    selects the champion model based on Macro F1, and persists production artifacts.
    """
    csv_file = data_path or get_data_path()
    out_dir = artifacts_dir or get_artifacts_dir()

    print("=" * 75)
    print("      MODEL TRAINING & SELECTION — SEVERITY PREDICTION PIPELINE")
    print("=" * 75)

    if not csv_file.exists():
        raise FileNotFoundError(f"Training dataset not found at {csv_file}")

    # 1. Load Raw Data
    df_raw = pd.read_csv(csv_file)
    logger.info("Loaded raw dataset with %d rows and %d columns", len(df_raw), len(df_raw.columns))

    # 2. Data Cleaning & ID Column Removal
    df_clean, identified_target, dropped_ids = clean_dataset(df_raw, target_col=target_col)
    logger.info("Target column identified: '%s'", identified_target)

    # 3. Feature Engineering (deterministic & leak-free)
    df_featured = engineer_features(df_clean)
    logger.info("Applied feature engineering. Feature count: %d", len(df_featured.columns) - 1)

    # 4. Train / Test Split (Stratified to respect class imbalance)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = prepare_train_test_split(
        df_featured,
        target_col=identified_target,
        test_size=0.2,
        random_state=42,
    )

    # 5. Fit Preprocessing ONLY on training data to prevent leakage
    preprocessor = DataPreprocessor()
    preprocessor.fit(X_train_raw, y_train_raw)

    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    y_train = preprocessor.transform_target(y_train_raw)
    y_test = preprocessor.transform_target(y_test_raw)

    class_names = preprocessor.classes_
    n_classes = len(class_names)
    logger.info("Transformed feature matrix: %d features, %d target classes: %s",
                X_train.shape[1], n_classes, class_names)

    # 6. Candidate Model Pool
    candidate_models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            C=1.0,
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_split=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        ),
    }

    results: Dict[str, Dict[str, Any]] = {}
    best_model_name = None
    best_macro_f1 = -1.0
    best_model_obj = None

    print("\nEvaluating Candidate Models...")
    print(f"{'Model Name':<22} | {'Accuracy':<9} | {'Macro F1':<9} | {'Weighted F1':<11} | {'ROC-AUC':<9}")
    print("-" * 75)

    for name, model in candidate_models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t0

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

        # Multiclass ROC-AUC (One-vs-Rest)
        roc_auc = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
                if n_classes == 2:
                    roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
                else:
                    roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
            except Exception as e:
                logger.warning("ROC-AUC computation skipped for %s: %s", name, e)

        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "macro_precision": round(float(macro_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else "N/A",
            "fit_time_seconds": round(float(fit_time), 4),
            "confusion_matrix": cm,
        }

        auc_display = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
        print(f"{name:<22} | {acc:.4f}    | {macro_f1:.4f}   | {weighted_f1:.4f}      | {auc_display}")

        # Champion selection: Macro F1 ensures high penalization for poor minority class detection
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model_name = name
            best_model_obj = model

    print("-" * 75)
    print(f"\nChampion Model Selected: '{best_model_name}' (Macro F1 = {best_macro_f1:.4f})")

    # Generate detailed classification report for champion
    y_best_pred = best_model_obj.predict(X_test)
    report_dict = classification_report(
        y_test,
        y_best_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # 7. Extract Feature Importances if available
    feature_importances = {}
    if hasattr(best_model_obj, "feature_importances_"):
        try:
            feat_names = preprocessor.get_feature_names_out()
            if len(feat_names) == len(best_model_obj.feature_importances_):
                importances = best_model_obj.feature_importances_
                top_indices = np.argsort(importances)[::-1][:15]
                for idx in top_indices:
                    feature_importances[feat_names[idx]] = round(float(importances[idx]), 4)
        except Exception as e:
            logger.warning("Feature importance extraction warning: %s", e)

    # 8. Compile Metadata
    metadata = {
        "project_title": "Severity Prediction Project",
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_info": {
            "file_name": csv_file.name,
            "total_rows_raw": int(len(df_raw)),
            "total_rows_cleaned": int(len(df_clean)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "target_column": identified_target,
            "dropped_id_columns": dropped_ids,
        },
        "target_classes": class_names,
        "input_features": preprocessor.feature_names_in_,
        "feature_schema": preprocessor.feature_schema_,
        "selected_model": best_model_name,
        "selected_model_metrics": results[best_model_name],
        "all_models_evaluated": results,
        "classification_report": report_dict,
        "top_feature_importances": feature_importances,
    }

    # 9. Save Artifacts
    save_model_artifacts(
        model=best_model_obj,
        preprocessor=preprocessor,
        metadata=metadata,
        artifacts_dir=out_dir,
    )

    print(f"\nModel artifacts successfully saved to: {out_dir}")
    print(f"  - model.pkl ({best_model_name})")
    print(f"  - preprocessing.pkl (DataPreprocessor)")
    print(f"  - metadata.json (Full training evaluation & schema)")
    print("=" * 75)

    return metadata


if __name__ == "__main__":
    train_and_evaluate_models()
