from pyspark.sql import SparkSession
from datetime import datetime

# Constants
DATE = '{{ ds }}'

spark = SparkSession.builder.appName("ExtractData").getOrCreate()

# Read data from database
query = "(SELECT * FROM data_credit) as data"
df = spark.read.jdbc(
    url="jdbc:postgresql://sources:5432/postgres",
    table=query,
    properties={
        "user": "postgres",
        "password": "postgres",
        "driver": "org.postgresql.Driver"
    }
)

# Simpan ke MinIO dalam format Parquet (data harian)
date_str = datetime.today().strftime('%Y-%m-%d')
df.write.mode("overwrite").parquet(f"s3a://ml-bucket/daily_data/data_credit_{date_str}.parquet")

spark.stop()