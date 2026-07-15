import pickle
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from MLOps.app.api.model_service import ModelInputError, PredictionService


class StubModel:
    def __init__(self, risk_score: float = 0.7) -> None:
        self.risk_score = risk_score

    def predict_proba(self, _features):
        return np.array([[1 - self.risk_score, self.risk_score]])


class PredictionServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.model_path = Path(self.temporary_directory.name) / "lightgbm_abt.pkl"

        artifact = {
            "model": StubModel(),
            "features": ["age", "occupation_type"],
            "decision_threshold": 0.5,
            "metrics": {"roc_auc": 0.75},
            "categorical_features": ["occupation_type"],
            "categories": {"occupation_type": ["Laborers", "Managers"]},
        }
        with self.model_path.open("wb") as file:
            pickle.dump(artifact, file)

        self.service = PredictionService(self.model_path)
        self.service.load()
        self.features = {"age": 35, "occupation_type": "Laborers"}

    def test_prediction_is_valid(self) -> None:
        score, predicted_class = self.service.predict(self.features)
        self.assertEqual(score, 0.7)
        self.assertEqual(predicted_class, 1)

    def test_legacy_features_key_is_normalized_on_load(self) -> None:
        self.assertEqual(self.service.expected_features, ["age", "occupation_type"])

    def test_loaded_model_is_current_while_artifact_exists(self) -> None:
        self.assertTrue(self.service.artifact_available)
        self.assertTrue(self.service.is_current)

    def test_missing_features_are_rejected(self) -> None:
        with self.assertRaises(ModelInputError) as context:
            self.service.predict({"age": 35})
        self.assertEqual(context.exception.missing_features, ["occupation_type"])

    def test_missing_artifact_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            PredictionService(self.model_path.with_name("missing.pkl")).load()

    def test_predict_before_load_is_rejected(self) -> None:
        service = PredictionService(self.model_path)
        with self.assertRaises(RuntimeError):
            service.predict(self.features)

    def test_removed_artifact_invalidates_loaded_model(self) -> None:
        self.model_path.unlink()

        self.assertTrue(self.service.is_loaded)
        self.assertFalse(self.service.artifact_available)
        self.assertFalse(self.service.is_current)

    def test_changed_artifact_is_reloaded(self) -> None:
        with self.model_path.open("rb") as file:
            artifact = pickle.load(file)
        artifact["model"] = StubModel(risk_score=0.4)
        artifact["artifact_revision"] = "new-version-with-different-size"
        with self.model_path.open("wb") as file:
            pickle.dump(artifact, file)

        self.assertFalse(self.service.is_current)
        self.service.load_if_needed()
        score, predicted_class = self.service.predict(self.features)

        self.assertTrue(self.service.is_current)
        self.assertEqual(score, 0.4)
        self.assertEqual(predicted_class, 0)


if __name__ == "__main__":
    unittest.main()
