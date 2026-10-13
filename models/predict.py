import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from .feature_engineering import engineer_features
from .model_utils import (
    get_artifacts_dir,
    get_severity_advice,
    load_model_artifacts,
)

logger = logging.getLogger("SeverityPrediction.Predict")


class SeverityPredictor:
    """
    Production inference engine for Severity Prediction.
    Loads persisted model, preprocessing pipeline, and metadata to ensure
    training-serving parity without data leakage or feature mismatch.
    """

    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = artifacts_dir or get_artifacts_dir()
        self.model, self.preprocessor, self.metadata = load_model_artifacts(self.artifacts_dir)
        self.classes: List[str] = self.metadata.get("target_classes", self.preprocessor.classes_)
        self.feature_schema: Dict[str, Any] = self.metadata.get("feature_schema", self.preprocessor.feature_schema_)
        logger.info(
            "SeverityPredictor initialized with model: %s, classes: %s",
            self.metadata.get("selected_model"),
            self.classes,
        )

    def validate_and_prepare_input(self, input_data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """
        Validate, cast types, and align input data against the training feature schema.
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}. Expected dict or pd.DataFrame.")

        # Align against feature schema
        for feat_name, schema in self.feature_schema.items():
            if feat_name not in df.columns:
                df[feat_name] = schema.get("default", 0 if schema.get("type") == "numerical" else "Unknown")
            else:
                if schema.get("type") == "numerical":
                    df[feat_name] = pd.to_numeric(df[feat_name], errors="coerce").fillna(schema.get("default", 0.0))
                else:
                    df[feat_name] = df[feat_name].astype(str)

        return df

    def predict(self, input_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Perform end-to-end severity inference on a single instance or batch:
          1. Validate and convert inputs to DataFrame
          2. Apply identical feature engineering
          3. Apply saved preprocessing pipeline (impute, scale, encode)
          4. Execute champion model inference
          5. Extract class probability confidence distribution
          6. Attach actionable domain guidance and risk triage advice
        """
        # 1. Prepare input
        df_input = self.validate_and_prepare_input(input_data)

        # 2. Feature engineering (identical to training)
        df_featured = engineer_features(df_input)

        # 3. Preprocessing transformation (using saved fitted pipeline)
        X_trans = self.preprocessor.transform(df_featured)

        # 4. Model Prediction
        y_pred_idx = self.model.predict(X_trans)
        predicted_classes = self.preprocessor.inverse_transform_target(y_pred_idx)
        top_prediction = str(predicted_classes[0])

        # 5. Probabilities / Confidence
        probabilities = {}
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_trans)[0]
            for class_name, prob in zip(self.classes, probs):
                probabilities[class_name] = round(float(prob * 100), 2)
            confidence = round(float(np.max(probs) * 100), 2)
        else:
            confidence = 100.0
            probabilities[top_prediction] = 100.0

        # 6. Contextual advice and risk classification
        advice = get_severity_advice(top_prediction)

        return {
            "prediction": top_prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "advice": advice,
            "selected_model": self.metadata.get("selected_model", "Champion Model"),
            "input_received": df_input.iloc[0].to_dict(),
        }


# Global singleton instance for high-throughput reuse
_cached_predictor: Optional[SeverityPredictor] = None


def predict_severity(input_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
    """Convenience function for instant prediction using cached model."""
    global _cached_predictor
    if _cached_predictor is None:
        _cached_predictor = SeverityPredictor()
    return _cached_predictor.predict(input_data)


if __name__ == "__main__":
    # Test sample prediction
    sample_incident = {
        "Driver_Age": 28,
        "Vehicle_Speed": 95,
        "Speed_Limit": 50,
        "Weather_Conditions": "Rainy",
        "Road_Surface_Conditions": "Wet",
        "Lighting_Conditions": "Darkness_No_Lights",
        "Vehicle_Type": "Motorcycle",
        "Driver_Alcohol": "Yes",
        "Seatbelt_Restraint": "Not_Worn",
        "Traffic_Density": "Medium",
        "Urban_Rural": "Rural",
        "Number_of_Vehicles": 2,
        "Number_of_Casualties": 2,
    }
    predictor = SeverityPredictor()
    result = predictor.predict(sample_incident)
    print("\n--- Test Inference Output ---")
    print(f"Predicted Severity: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Class Probabilities: {result['probabilities']}")
    print(f"Action: {result['advice']['action']}")
