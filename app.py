import logging
import os
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_from_directory

from models.model_utils import get_artifacts_dir, get_outputs_dir, get_project_root
from models.predict import SeverityPredictor
from models.train_model import train_and_evaluate_models

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("SeverityPrediction.App")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "severity-prediction-production-key-2026")

# Lazy-loaded predictor
_predictor = None


def get_predictor() -> SeverityPredictor:
    """Ensure artifacts exist and return singleton SeverityPredictor instance."""
    global _predictor
    artifacts_dir = get_artifacts_dir()
    model_path = artifacts_dir / "model.pkl"
    meta_path = artifacts_dir / "metadata.json"

    if not model_path.exists() or not meta_path.exists():
        logger.info("Artifacts not found. Training baseline model before launching service...")
        train_and_evaluate_models()

    if _predictor is None:
        _predictor = SeverityPredictor(artifacts_dir=artifacts_dir)
    return _predictor


@app.route("/", methods=["GET"])
def index():
    """Render home portal displaying the dynamically configured severity prediction form."""
    try:
        predictor = get_predictor()
        schema = predictor.feature_schema
        selected_model = predictor.metadata.get("selected_model", "Trained ML Model")
        target_classes = predictor.classes
        model_metrics = predictor.metadata.get("selected_model_metrics", {})
        return render_template(
            "index.html",
            feature_schema=schema,
            selected_model=selected_model,
            target_classes=target_classes,
            model_metrics=model_metrics,
        )
    except Exception as e:
        logger.error("Error loading index page: %s", e, exc_info=True)
        return render_template(
            "index.html",
            feature_schema={},
            selected_model="Unavailable",
            target_classes=[],
            model_metrics={},
            error=f"Model initialization error: {str(e)}",
        )


@app.route("/predict", methods=["POST"])
def predict():
    """Handle web form submission for severity prediction."""
    try:
        predictor = get_predictor()
        # Parse inputs from form
        input_data = {}
        for feat_name, schema in predictor.feature_schema.items():
            val = request.form.get(feat_name)
            if val is None or val.strip() == "":
                # Fallback to schema default
                input_data[feat_name] = schema.get("default")
            else:
                if schema.get("type") == "numerical":
                    try:
                        input_data[feat_name] = float(val)
                    except ValueError:
                        input_data[feat_name] = schema.get("default", 0.0)
                else:
                    input_data[feat_name] = val.strip()

        # Execute leak-free inference
        inference_result = predictor.predict(input_data)

        return render_template(
            "result.html",
            prediction=inference_result["prediction"],
            confidence=inference_result["confidence"],
            probabilities=inference_result["probabilities"],
            advice=inference_result["advice"],
            selected_model=inference_result["selected_model"],
            input_received=inference_result["input_received"],
        )

    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        predictor = get_predictor()
        return render_template(
            "index.html",
            feature_schema=predictor.feature_schema,
            selected_model=predictor.metadata.get("selected_model", "Trained ML Model"),
            target_classes=predictor.classes,
            model_metrics=predictor.metadata.get("selected_model_metrics", {}),
            error=f"Prediction encountered an error: {str(e)}",
        )


@app.route("/about", methods=["GET"])
def about():
    """Render technical architecture, ML lifecycle modules M1-M5, model comparison, and EDA gallery."""
    try:
        predictor = get_predictor()
        metadata = predictor.metadata
        all_models = metadata.get("all_models_evaluated", {})
        selected_model = metadata.get("selected_model", "Trained Model")
        metrics = metadata.get("selected_model_metrics", {})
        
        # Collect generated EDA charts if present
        eda_dir = get_outputs_dir()
        eda_charts = []
        if eda_dir.exists():
            for p in sorted(eda_dir.glob("*.png")):
                eda_charts.append({
                    "filename": p.name,
                    "title": p.stem.replace("_", " ").title(),
                })

        return render_template(
            "about.html",
            all_models=all_models,
            selected_model=selected_model,
            metrics=metrics,
            metadata=metadata,
            eda_charts=eda_charts,
        )
    except Exception as e:
        logger.error("Error loading about page: %s", e)
        return render_template(
            "about.html",
            all_models={},
            selected_model="Unavailable",
            metrics={},
            metadata={},
            eda_charts=[],
        )


@app.route("/outputs/eda/<path:filename>")
def serve_eda_file(filename):
    """Serve generated exploratory data analysis visualizations."""
    return send_from_directory(get_outputs_dir(), filename)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic inference requests."""
    try:
        if not request.is_json:
            return jsonify({"status": "error", "message": "Expected JSON payload with Content-Type: application/json"}), 400

        payload = request.get_json()
        predictor = get_predictor()
        result = predictor.predict(payload)
        return jsonify({
            "status": "success",
            "prediction": result["prediction"],
            "confidence_percentage": result["confidence"],
            "class_probabilities": result["probabilities"],
            "action_advice": result["advice"],
            "model_used": result["selected_model"],
        }), 200

    except Exception as e:
        logger.error("API inference failed: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint for Docker and monitoring orchestrators."""
    try:
        predictor = get_predictor()
        return jsonify({
            "status": "healthy",
            "active_model": predictor.metadata.get("selected_model"),
            "target_classes": predictor.classes,
            "total_features": len(predictor.feature_schema),
        }), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
