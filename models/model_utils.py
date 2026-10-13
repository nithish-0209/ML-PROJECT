import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("SeverityPrediction")


def get_project_root() -> Path:
    """Return the absolute Path to the severity_prediction_project root directory."""
    return Path(__file__).resolve().parent.parent


def get_artifacts_dir() -> Path:
    """Return the artifacts directory path."""
    artifacts_dir = get_project_root() / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def get_outputs_dir() -> Path:
    """Return the outputs/eda directory path."""
    outputs_dir = get_project_root() / "outputs" / "eda"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def get_data_path() -> Path:
    """Return the default path to data/dataset.csv."""
    return get_project_root() / "data" / "dataset.csv"


def detect_target_column(df: pd.DataFrame, explicit_target: Optional[str] = None) -> str:
    """
    Intelligently identify the target/severity column in the DataFrame.
    Supports user override, common severity column aliases, and categorical detection.
    """
    if explicit_target and explicit_target in df.columns:
        return explicit_target

    candidate_names = [
        "accident_severity",
        "severity",
        "severity_level",
        "target",
        "injury_severity",
        "condition_severity",
        "crash_severity",
        "incident_severity",
        "class",
    ]

    lower_cols = {col.lower(): col for col in df.columns}
    for candidate in candidate_names:
        if candidate in lower_cols:
            return lower_cols[candidate]

    # Partial match
    for col in df.columns:
        if "severity" in col.lower():
            return col

    # Fallback to the last column if it is categorical / object
    last_col = df.columns[-1]
    if df[last_col].dtype == "object" or df[last_col].nunique() <= 10:
        logger.warning("Target column not found by name. Defaulting to last column: %s", last_col)
        return last_col

    raise ValueError(
        f"Unable to automatically identify target column in columns: {list(df.columns)}. "
        "Please specify target explicitly."
    )


def save_model_artifacts(
    model: Any,
    preprocessor: Any,
    metadata: Dict[str, Any],
    artifacts_dir: Optional[Path] = None,
) -> None:
    """Save trained model, preprocessor, and metadata JSON to artifacts directory."""
    target_dir = artifacts_dir or get_artifacts_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / "model.pkl"
    prep_path = target_dir / "preprocessing.pkl"
    meta_path = target_dir / "metadata.json"

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, prep_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved preprocessor to %s", prep_path)
    logger.info("Saved metadata to %s", meta_path)


def load_model_artifacts(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load trained model, preprocessor, and metadata JSON from artifacts directory."""
    target_dir = artifacts_dir or get_artifacts_dir()

    model_path = target_dir / "model.pkl"
    prep_path = target_dir / "preprocessing.pkl"
    meta_path = target_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Train the model first.")
    if not prep_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at {prep_path}.")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata artifact not found at {meta_path}.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, preprocessor, metadata


def get_severity_advice(severity: str) -> Dict[str, str]:
    """Provide contextual interpretation, risk grade, and safety recommendations."""
    severity_lower = str(severity).lower()

    if "fatal" in severity_lower or "critical" in severity_lower:
        return {
            "level": "Critical / Fatal",
            "badge_class": "badge-fatal",
            "color": "#dc2626",
            "urgency": "Immediate Emergency Response",
            "action": "Immediate dispatch of Advanced Life Support (ALS) paramedic units, heavy rescue extraction, and trauma center diversion.",
            "description": "Extreme trauma risk with life-threatening outcomes. High speed, lack of restraint, severe impact, or multiple casualties identified.",
        }
    elif "severe" in severity_lower or "serious" in severity_lower or "high" in severity_lower:
        return {
            "level": "Severe / Serious",
            "badge_class": "badge-severe",
            "color": "#ea580c",
            "urgency": "High Priority Dispatch",
            "action": "Urgent emergency medical response, traffic corridor containment, and hospital trauma team standby.",
            "description": "High probability of major injuries, structural vehicle deformation, and significant road safety hazard.",
        }
    elif "moderate" in severity_lower or "medium" in severity_lower:
        return {
            "level": "Moderate",
            "badge_class": "badge-moderate",
            "color": "#d97706",
            "urgency": "Standard Emergency Response",
            "action": "Deploy standard medical triage and traffic clearance units. Evaluate driver vitals and monitor traffic flow.",
            "description": "Moderate risk scenario. Non-life-threatening injuries likely, requiring medical evaluation and vehicle towing.",
        }
    else:
        return {
            "level": "Minor / Slight",
            "badge_class": "badge-minor",
            "color": "#16a34a",
            "urgency": "Low Priority / Standard Response",
            "action": "Routine roadside assistance, self-reporting documentation, and local traffic officer assistance.",
            "description": "Minimal injury probability. Primarily property damage or superficial scratches under controlled conditions.",
        }
