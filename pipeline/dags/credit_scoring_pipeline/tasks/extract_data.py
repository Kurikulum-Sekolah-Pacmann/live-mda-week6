from pyspark.sql import SparkSession
from datetime import datetime
from airflow.hooks.base import BaseHook

# Data Sources connections
POSTGRES_CONN = BaseHook.get_connection('credit-data-db-conn')
POSTGRES_USERNAME = POSTGRES_CONN.login
POSTGRES_PASSWORD = POSTGRES_CONN.password
POSTGRES_HOST = POSTGRES_CONN.host
POSTGRES_PORT = POSTGRES_CONN.port
POSTGRES_DATABASE = POSTGRES_CONN.schema

DATE = '{{ ds }}'

# Create Spark session
spark = SparkSession.builder.appName("ExtractData").getOrCreate()

# Read data from database
query = "(SELECT * FROM data_credit) as data"
df = spark.read.jdbc(
    url=f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}",
    table=query,
    properties={
        "user": POSTGRES_USERNAME,
        "password": POSTGRES_PASSWORD,
        "driver": "org.postgresql.Driver"
    }
)

# Simpan ke MinIO dalam format Parquet (data harian)
date_str = datetime.today().strftime('%Y-%m-%d')
df.write.mode("overwrite").parquet(f"s3a://ml-bucket/daily_data/data_credit_{date_str}.parquet")

spark.stop()