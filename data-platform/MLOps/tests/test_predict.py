import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np

from Model.predict import load_artifact, load_features_from_abt, predict_score


class PredictScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        model = Mock()
        model.predict_proba.return_value = np.array([[0.2, 0.8]])
        self.model = model
        self.artifact = {
            "model": model,
            "input_features": ["age", "occupation_type"],
            "decision_threshold": 0.5,
            "categorical_features": ["occupation_type"],
            "categories": {"occupation_type": ["Laborers", "Managers"]},
        }

    def test_predicts_dataframe_row(self) -> None:
        features = pd.DataFrame(
            [{"occupation_type": "Laborers", "age": "35", "ignored": 1}]
        )
        result = predict_score(features, self.artifact)

        self.assertEqual(result["risk_score"], 0.8)
        self.assertEqual(result["predicted_class"], 1)
        self.assertEqual(result["decision_threshold"], 0.5)
        self.assertEqual(result["decision"], "NEGAR_CREDITO")

        model_input = self.model.predict_proba.call_args.args[0]
        self.assertEqual(list(model_input.columns), ["age", "occupation_type"])
        self.assertEqual(str(model_input["occupation_type"].dtype), "category")

    def test_missing_feature_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "occupation_type"):
            predict_score(pd.DataFrame([{"age": 35}]), self.artifact)

    def test_load_artifact_rejects_missing_file(self) -> None:
        with TemporaryDirectory() as directory:
            missing_path = Path(directory) / "missing.pkl"
            with self.assertRaises(FileNotFoundError):
                load_artifact(missing_path)

    @patch("Model.predict.pd.read_sql")
    @patch("Model.predict.get_database_connection")
    def test_loads_customer_from_abt(self, get_connection, read_sql) -> None:
        connection = Mock()
        get_connection.return_value = connection
        read_sql.return_value = pd.DataFrame([{"sk_id_curr": 100002, "age": 35}])

        result = load_features_from_abt(100002, "postgres_data_db", ["age"])

        self.assertEqual(result.iloc[0]["sk_id_curr"], 100002)
        self.assertIn("sk_id_curr = 100002", read_sql.call_args.args[0])
        connection.close.assert_called_once_with()

    @patch("Model.predict.pd.read_sql", return_value=pd.DataFrame())
    @patch("Model.predict.get_database_connection")
    def test_unknown_customer_is_rejected(self, get_connection, _read_sql) -> None:
        get_database_connection = get_connection.return_value
        with self.assertRaisesRegex(ValueError, "não encontrado"):
            load_features_from_abt(999999, "postgres_data_db", ["age"])
        get_database_connection.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
