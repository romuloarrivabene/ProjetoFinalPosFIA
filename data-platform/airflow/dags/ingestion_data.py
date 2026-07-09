from datetime import datetime
import sys

from airflow import DAG
from airflow.decorators import task

sys.path.append("/opt/airflow/DataPipeline")

from ingestion import run_csv_ingestion


CONN_ID = "postgres_data_db"
PASTA_DATA = "/opt/airflow/data/csv"

with DAG(
    dag_id="loadfile_csv_to_postgres",
    start_date=datetime(2026, 6, 24),
    schedule=None,
    catchup=False,
    tags=["ingestion", "postgres"],
) as dag:

    @task(task_id="loadfile")
    def loadfile(conn_id: str, source_dir: str):
        run_csv_ingestion(conn_id, source_dir)

    loadfile(conn_id=CONN_ID, source_dir=PASTA_DATA)
