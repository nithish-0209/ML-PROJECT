import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import app


class TestSeverityApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_01_index_page(self):
        """Test GET / returns 200 and renders form fields."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")
        self.assertIn("Severity Prediction Project", html)
        self.assertIn("Incident Assessment Parameters", html)
        self.assertIn("<form", html)
        self.assertIn("Active Model Specification", html)
        print("[PASS] GET / passed")

    def test_02_about_page(self):
        """Test GET /about returns 200 and covers M1-M5."""
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")
        self.assertIn("System Architecture", html)
        self.assertIn("Supervised Learning Benchmark Scorecard", html)
        self.assertIn("M1 — Machine Learning Lifecycle", html)
        self.assertIn("M2 — Supervised Learning", html)
        self.assertIn("M3 — Tree-Based Ensembles", html)
        self.assertIn("M4 — Feature Engineering", html)
        self.assertIn("M5 — Model Evaluation", html)
        print("[PASS] GET /about passed")

    def test_03_health_api(self):
        """Test GET /api/health returns healthy status."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("target_classes", data)
        print("[PASS] GET /api/health passed")

    def test_04_predict_post_form(self):
        """Test POST /predict with form data renders result page."""
        form_payload = {
            "Driver_Age": "25",
            "Vehicle_Speed": "90",
            "Speed_Limit": "50",
            "Weather_Conditions": "Rainy",
            "Road_Surface_Conditions": "Wet",
            "Lighting_Conditions": "Darkness_No_Lights",
            "Vehicle_Type": "Motorcycle",
            "Driver_Alcohol": "Yes",
            "Seatbelt_Restraint": "Not_Worn",
            "Traffic_Density": "Medium",
            "Urban_Rural": "Rural",
            "Number_of_Vehicles": "2",
            "Number_of_Casualties": "2",
        }
        response = self.client.post("/predict", data=form_payload)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")
        self.assertIn("Triage Classification Result", html)
        self.assertIn("Automated Triage Protocol", html)
        self.assertIn("Submitted Incident Attributes", html)
        self.assertIn("Assess Another Incident", html)
        print("[PASS] POST /predict (Form) passed")

    def test_05_predict_post_api(self):
        """Test POST /api/predict with JSON returns valid prediction."""
        json_payload = {
            "Driver_Age": 45,
            "Vehicle_Speed": 35,
            "Speed_Limit": 50,
            "Weather_Conditions": "Clear",
            "Road_Surface_Conditions": "Dry",
            "Lighting_Conditions": "Daylight",
            "Vehicle_Type": "Sedan",
            "Driver_Alcohol": "No",
            "Seatbelt_Restraint": "Worn",
            "Traffic_Density": "Low",
            "Urban_Rural": "Urban",
            "Number_of_Vehicles": 1,
            "Number_of_Casualties": 1,
        }
        response = self.client.post("/api/predict", json=json_payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("prediction", data)
        self.assertIn("confidence_percentage", data)
        self.assertIn("class_probabilities", data)
        self.assertIn("action_advice", data)
        print(f"[PASS] POST /api/predict (JSON) passed: Result = {data['prediction']}, Confidence = {data['confidence_percentage']}%")


if __name__ == "__main__":
    unittest.main()
