import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from .model_utils import detect_target_column

logger = logging.getLogger("SeverityPrediction.Preprocessing")


def clean_dataset(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    drop_duplicates: bool = True,
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Clean raw dataset before modeling:
      - Detect target column
      - Remove duplicate rows
      - Identify and drop primary ID/Index columns to prevent leakage and noise
      - Drop rows with null target values
    """
    df_clean = df.copy()

    # Identify target
    target = target_col or detect_target_column(df_clean)

    # Remove duplicates
    if drop_duplicates:
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        dropped = initial_len - len(df_clean)
        if dropped > 0:
            logger.info("Dropped %d duplicate rows (initial: %d, remaining: %d)", dropped, initial_len, len(df_clean))

    # Drop target nulls if any
    if target in df_clean.columns:
        null_targets = df_clean[target].isnull().sum()
        if null_targets > 0:
            logger.warning("Dropping %d rows with missing target '%s'", null_targets, target)
            df_clean = df_clean.dropna(subset=[target])

    # Identify and drop ID columns
    id_cols = []
    for col in df_clean.columns:
        if col == target:
            continue
        c_lower = col.lower()
        is_id_name = (
            c_lower.endswith("_id")
            or c_lower.endswith("_index")
            or c_lower in ["id", "index", "accident_index", "case_id", "incident_id"]
        )
        is_unique_string = (
            (df_clean[col].dtype == "object" or str(df_clean[col].dtype).startswith("str"))
            and df_clean[col].nunique() == len(df_clean)
            and len(df_clean) > 50
        )
        if is_id_name or is_unique_string:
            id_cols.append(col)

    if id_cols:
        logger.info("Identified and excluded ID columns to prevent leakage: %s", id_cols)
        df_clean = df_clean.drop(columns=id_cols)

    return df_clean, target, id_cols


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Robust, production-grade Preprocessing Pipeline.
    Prevents data leakage by fitting exclusively on training data.
    Provides identical transformations for training and serving inference.
    """

    def __init__(self):
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.feature_names_in_: List[str] = []
        self.target_col: Optional[str] = None
        self.column_transformer: Optional[ColumnTransformer] = None
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.classes_: List[str] = []
        self.feature_schema_: Dict[str, Dict[str, Any]] = {}
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataPreprocessor":
        """Fit preprocessor pipelines on training features and targets."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = list(X.columns)

        # Distinguish numerical and categorical columns
        self.numerical_cols = []
        self.categorical_cols = []

        for col in self.feature_names_in_:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.numerical_cols.append(col)
            else:
                self.categorical_cols.append(col)

        logger.info(
            "Identified %d numerical features and %d categorical features",
            len(self.numerical_cols),
            len(self.categorical_cols),
        )

        # Build feature schema for validation and dynamic UI generation
        self.feature_schema_ = {}
        for col in self.numerical_cols:
            col_series = pd.to_numeric(X[col], errors="coerce").dropna()
            self.feature_schema_[col] = {
                "type": "numerical",
                "min": float(col_series.min()) if len(col_series) else 0.0,
                "max": float(col_series.max()) if len(col_series) else 100.0,
                "mean": float(col_series.mean()) if len(col_series) else 50.0,
                "median": float(col_series.median()) if len(col_series) else 50.0,
                "default": float(col_series.median()) if len(col_series) else 0.0,
            }

        for col in self.categorical_cols:
            categories = [str(val) for val in X[col].dropna().unique() if str(val).strip() != ""]
            top_val = str(X[col].mode().iloc[0]) if not X[col].dropna().empty else (categories[0] if categories else "")
            self.feature_schema_[col] = {
                "type": "categorical",
                "categories": sorted(categories),
                "default": top_val,
            }

        # Build transformers
        transformers = []
        if self.numerical_cols:
            num_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("num", num_pipe, self.numerical_cols))

        if self.categorical_cols:
            cat_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
            transformers.append(("cat", cat_pipe, self.categorical_cols))

        self.column_transformer = ColumnTransformer(transformers=transformers, remainder="drop")
        self.column_transformer.fit(X)

        # Fit target label encoder if provided
        if y is not None:
            y_clean = y.dropna().astype(str)
            self.label_encoder.fit(y_clean)
            self.classes_ = list(self.label_encoder.classes_)
            logger.info("Target classes encoded: %s", self.classes_)

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted imputers, scalers, and encoders."""
        if not self.is_fitted or self.column_transformer is None:
            raise RuntimeError("DataPreprocessor must be fitted before transforming data.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Ensure all expected columns exist (fill missing columns with default values from schema)
        X_aligned = X.copy()
        for col in self.feature_names_in_:
            if col not in X_aligned.columns:
                default_val = self.feature_schema_.get(col, {}).get("default", 0 if col in self.numerical_cols else "Unknown")
                X_aligned[col] = default_val

        # Select only expected features in the exact fitted order
        X_aligned = X_aligned[self.feature_names_in_]

        return self.column_transformer.transform(X_aligned)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def transform_target(self, y: pd.Series) -> np.ndarray:
        """Encode target labels to integer indices."""
        return self.label_encoder.transform(y.astype(str))

    def inverse_transform_target(self, y_indices: np.ndarray) -> np.ndarray:
        """Decode integer indices back to original string target labels."""
        return self.label_encoder.inverse_transform(y_indices)

    def get_feature_names_out(self) -> List[str]:
        """Get transformed feature names after scaling and one-hot encoding."""
        if self.column_transformer is None:
            return []
        try:
            return list(self.column_transformer.get_feature_names_out())
        except Exception:
            return [f"feature_{i}" for i in range(len(self.numerical_cols) + 10)]


def prepare_train_test_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into train and test sets using stratified sampling to preserve class distribution.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str)

    # Check for stratification feasibility
    min_class_count = y.value_counts().min()
    stratify = y if min_class_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    logger.info(
        "Train set shape: %s, Test set shape: %s (Stratified: %s)",
        X_train.shape,
        X_test.shape,
        stratify is not None,
    )
    return X_train, X_test, y_train, y_test
