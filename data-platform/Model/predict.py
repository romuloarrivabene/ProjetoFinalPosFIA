"""
predict.py — Serviço de predição local de risco de crédito.

Lê o artefato `.pkl` gerado na pipeline, recebe os dados de um cliente
(via ID do banco ou via JSON local) e retorna o escore de risco e a decisão.
"""
import argparse
import pickle
from typing import Any

import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = MODEL_DIR / "artifacts/lightgbm_abt.pkl"


def get_database_connection(conn_id: str = "postgres_data_db", silent: bool = False):
    """Retorna uma conexão ativa com o banco.

    Detecta automaticamente se está rodando dentro do fluxo do Airflow (usa
    PostgresHook) ou de forma isolada (usa SQLAlchemy).
    """
    from sqlalchemy import create_engine
    import os

    # Mapeando o host baseado no ambiente (se roda dentro do ecossistema docker ou na máquina local)
    # No docker o host do banco chama-se 'postgres'. Na máquina local acessamos via 'localhost'
    host = "postgres" if os.path.exists("/.dockerenv") else "localhost"
    conn_str = f"postgresql://airflow:airflow@{host}:5432/data"

    if not silent:
        print(f"[CONEXÃO] Execução isolada detectada (Local/Notebook). Conectando via SQLAlchemy em '{host}'.")
    engine = create_engine(conn_str)
    return engine.raw_connection()

def load_artifact(artifact_path: Path) -> dict[str, Any]:
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artefato não encontrado: {artifact_path}. Execute o train.py antes.")
    with open(artifact_path, "rb") as f:
        return pickle.load(f)

def load_features_from_abt(
    sk_id: int,
    conn_id: str,
    features_esperadas: list[str],
) -> pd.DataFrame:
    """Busca os dados fresquinhos do cliente direto na ABT usando o utils."""
    conn = get_database_connection(conn_id, silent=True)
    
    # Puxa o cliente, excluindo colunas não preditivas
    query = f'SELECT * FROM application_abt WHERE sk_id_curr = {sk_id} LIMIT 1;'
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        raise ValueError(f"Cliente sk_id_curr={sk_id} não encontrado na base de dados (ABT).")
        
    return df

def predict_score(
    df_features: pd.DataFrame,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    """Executa a inferência e retorna a decisão de negócio."""
    model = artifact["model"]
    features_esperadas = artifact.get("input_features", artifact.get("features"))
    if not features_esperadas:
        raise ValueError("Artefato inválido: lista de features ausente.")

    missing_features = sorted(set(features_esperadas).difference(df_features.columns))
    if missing_features:
        raise ValueError(f"Features obrigatórias ausentes: {', '.join(missing_features)}")

    threshold = float(artifact["decision_threshold"])
    df_inf = df_features.reindex(columns=features_esperadas).copy()

    categorical_features = set(artifact.get("categorical_features", []))
    saved_categories = artifact.get("categories", {})
    for column in df_inf.columns:
        if column in categorical_features:
            df_inf[column] = pd.Categorical(
                df_inf[column],
                categories=saved_categories.get(column),
            )
        else:
            df_inf[column] = pd.to_numeric(df_inf[column], errors="coerce")

    proba = float(model.predict_proba(df_inf)[0, 1])

    return {
        "risk_score": round(proba, 4),
        "decision_threshold": threshold,
        "predicted_class": int(proba >= threshold),
        "decision": "NEGAR_CREDITO" if proba >= threshold else "APROVAR_CREDITO"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia de Risco de Crédito")
    parser.add_argument("--sk-id", type=int, required=True, help="ID do cliente para busca na ABT")
    args = parser.parse_args()

    print(f"[PREDICT] Carregando artefato...")
    print(ARTIFACT_PATH)
    artifact = load_artifact(ARTIFACT_PATH)
    artifact_features = artifact.get("input_features", artifact.get("features"))
    if not artifact_features:
        raise ValueError("Artefato inválido: lista de features ausente.")
    
    print(f"[PREDICT] Buscando features para sk_id_curr = {args.sk_id}...")
    df_client = load_features_from_abt(
        args.sk_id,
        "postgres_data_db",
        artifact_features,
    )
    
    print("[PREDICT] Rodando modelo...")
    resultado = predict_score(df_client, artifact)
    
    print("\n" + "="*40)
    print(" RESULTADO DA ANÁLISE DE CRÉDITO ")
    print("="*40)
    print(f"  > Cliente ID    : {args.sk_id}")
    print(f"  > Risk Score    : {resultado['risk_score']}")
    print(f"  > Ponto de Corte: {resultado['decision_threshold']}")
    print(f"  > DECISÃO FINAL : {resultado['decision']}")
    print("="*40 + "\n")
