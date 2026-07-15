import json
import pickle
import unittest
from pathlib import Path


DATA_PLATFORM_DIR = Path(__file__).resolve().parents[2]


class ConfigurationTest(unittest.TestCase):
    def test_required_delivery_structure_exists(self) -> None:
        required_paths = (
            "Dados/README.md",
            "DataPipeline/data_sanitization.py",
            "DataPipeline/abt_transform.py",
            "DataPipeline/exp_analysis.ipynb",
            "DataPipeline/config_pipeline.json",
            "Model/train.py",
            "Model/config_model.json",
            "Model/evaluation.ipynb",
            "Model/predict.py",
            "MLOps/README.md",
            "MLOps/Dockerfile.api",
            "MLOps/Dockerfile.frontend",
            "MLOps/app/api/main.py",
            "MLOps/app/frontend/app.py",
            "airflow/dags/pipeline_orchestration.py",
            "docker-compose.yml",
            "requirements.txt",
        )
        missing = [
            path for path in required_paths if not (DATA_PLATFORM_DIR / path).is_file()
        ]
        self.assertEqual(missing, [])
        self.assertTrue((DATA_PLATFORM_DIR.parent / "README.md").is_file())

    def test_pipeline_configuration_has_required_sections(self) -> None:
        config = json.loads(
            (DATA_PLATFORM_DIR / "DataPipeline/config_pipeline.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(
            {"ingestion_table", "database", "indexes", "sanitization"}
            <= config.keys()
        )

    def test_model_configuration_has_required_sections(self) -> None:
        config = json.loads(
            (DATA_PLATFORM_DIR / "Model/config_model.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(
            {"metadata", "database", "variables", "parameters", "validation"}
            <= config.keys()
        )

    def test_model_features_match_persisted_artifact(self) -> None:
        config = json.loads(
            (DATA_PLATFORM_DIR / "Model/config_model.json").read_text(
                encoding="utf-8"
            )
        )
        artifact_path = DATA_PLATFORM_DIR / "Model/artifacts/lightgbm_abt.pkl"
        if not artifact_path.is_file():
            self.skipTest("Artefato será gerado pelo pipeline antes da demonstração.")

        with artifact_path.open("rb") as file:
            artifact = pickle.load(file)
        self.assertEqual(
            config["variables"]["input_features"],
            artifact.get("input_features", artifact.get("features")),
        )
        self.assertEqual(
            config["variables"]["categorical_features"],
            artifact["categorical_features"],
        )


if __name__ == "__main__":
    unittest.main()
