from airflow.decorators import dag
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta

# Define the list of JAR files required for Spark
jar_list = [
    '/opt/spark/jars/hadoop-aws-3.3.1.jar',
    '/opt/spark/jars/aws-java-sdk-bundle-1.11.901.jar',
    '/opt/spark/jars/postgresql-42.2.23.jar'
]

# Define Spark configuration
spark_conf = {
    'spark.hadoop.fs.s3a.access.key': 'minio',
    'spark.hadoop.fs.s3a.secret.key': 'minio123',
    'spark.hadoop.fs.s3a.endpoint': 'http://minio:9000',
    'spark.hadoop.fs.s3a.path.style.access': 'true',
    'spark.hadoop.fs.s3a.impl': 'org.apache.hadoop.fs.s3a.S3AFileSystem',
    'spark.dynamicAllocation.enabled': 'true',
    'spark.dynamicAllocation.maxExecutors': '3',
    'spark.dynamicAllocation.minExecutors': '1',
    'spark.dynamicAllocation.initialExecutors': '1',
    'spark.executor.memory': '4g',  # Define RAM per executor
    'spark.executor.cores': '2',  # Define cores per executor
    'spark.scheduler.mode': 'FAIR'
}

@dag(
    dag_id="ml_pipeline_credit_scoring",
    start_date=datetime(2024, 9, 1),
    schedule="@daily",
    catchup=False,
    tags=["dellstore"],
    description="Extract, and Load Dellstore data into Staging Area"
)

def ml_pipeline_credit_scoring():
    extract_data = SparkSubmitOperator(
        task_id="extract_data_postgres",
        application="/opt/airflow/dags/credit_scoring_pipeline/tasks/extract_data.py",
        conn_id="spark-conn",
        conf=spark_conf,
        jars=','.join(jar_list)
    )
    
    preprocessing = SparkSubmitOperator(
        task_id="preprocessing",
        application="/opt/airflow/dags/credit_scoring_pipeline/tasks/preprocess.py",
        conn_id="spark-conn",
        conf=spark_conf,
        jars=','.join(jar_list)
    )
    
    training = SparkSubmitOperator(
        task_id="training",
        application="/opt/airflow/dags/credit_scoring_pipeline/tasks/train.py",
        conn_id="spark-conn",
        conf=spark_conf,
        jars=','.join(jar_list)
    )
    
    evaluate = SparkSubmitOperator(
        task_id="evaluate",
        application="/opt/airflow/dags/credit_scoring_pipeline/tasks/evaluate.py",
        conn_id="spark-conn",
        conf=spark_conf,
        jars=','.join(jar_list)
    )

    extract_data >> preprocessing >> training >> evaluate

ml_pipeline_credit_scoring()