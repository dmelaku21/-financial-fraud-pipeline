from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


default_args = {
    "owner": "fraud-data-engineering",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="financial_fraud_pipeline",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["fraud", "machine-learning", "data-pipeline"],
) as dag:

    batch_ingestion = BashOperator(
        task_id="batch_ingestion",
        bash_command=(
            "cd /opt/airflow/project && "
            "python src/ingestion/batch_ingestion.py"
        ),
    )

    stream_simulation = BashOperator(
        task_id="stream_simulation",
        bash_command=(
            "cd /opt/airflow/project && "
            "python src/ingestion/stream_simulator.py"
        ),
    )

    clean_data = BashOperator(
        task_id="clean_data",
        bash_command=(
            "cd /opt/airflow/project && "
            "python src/processing/clean_transactions.py"
        ),
    )

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=(
            "cd /opt/airflow/project && "
            "python src/features/build_features.py"
        ),
    )

    quality_validation = BashOperator(
        task_id="quality_validation",
        bash_command=(
            "cd /opt/airflow/project && "
            "python src/validation/gx_validation.py"
        ),
    )

    spark_processing = BashOperator(
        task_id="spark_processing",
        bash_command=(
            "cd /opt/airflow/project && "
            "python src/processing/spark_processing.py"
        ),
    )

    generate_reports = BashOperator(
        task_id="generate_reports",
        bash_command=(
            "cd /opt/airflow/project && "
            "python src/reporting/generate_reports.py"
        ),
    )

    load_ml_ready = BashOperator(
        task_id="load_ml_ready",
        bash_command=(
            "cd /opt/airflow/project && "
            "python src/ml/create_ml_ready.py"
        ),
    )

    (
        batch_ingestion
        >> stream_simulation
        >> clean_data
        >> feature_engineering
        >> quality_validation
        >> spark_processing
        >> generate_reports
        >> load_ml_ready
    )