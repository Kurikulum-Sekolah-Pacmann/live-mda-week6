from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler
import boto3

# Initialize Spark Session
spark = SparkSession.builder.appName("Preprocessing").getOrCreate()

# MinIO Configuration
s3_endpoint = "http://minio:9000"
s3_access_key = "minio"
s3_secret_key = "minio123"
bucket_name = "ml-bucket"
data_prefix = "daily_data/"

# Initialize Boto3 client
s3 = boto3.client(
    's3',
    endpoint_url=s3_endpoint,
    aws_access_key_id=s3_access_key,
    aws_secret_access_key=s3_secret_key
)

# List all Parquet files in the MinIO bucket
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=data_prefix)

# Extract file paths
file_list = [
    f"s3a://{bucket_name}/{obj['Key']}" for obj in response.get('Contents', []) if obj['Key'].endswith(".parquet")
]

# Check if there are files to process
if not file_list:
    print("No new data found in MinIO. Skipping preprocessing.")
    spark.stop()
    exit(0)

# Load and merge all daily data
df_combined = spark.read.parquet(*file_list)

print(df_combined.show())

# Handle missing values
df = df_combined.fillna({'person_emp_length': 0, 'loan_int_rate': df_combined.agg({'loan_int_rate': 'mean'}).collect()[0][0]})

# Encode categorical variables
indexers = [
    StringIndexer(inputCol="person_home_ownership", outputCol="person_home_ownership_index"),
    StringIndexer(inputCol="loan_intent", outputCol="loan_intent_index"),
    StringIndexer(inputCol="loan_grade", outputCol="loan_grade_index"),
    StringIndexer(inputCol="cb_person_default_on_file", outputCol="cb_person_default_on_file_index"),
]

for indexer in indexers:
    df = indexer.fit(df).transform(df)

# Assemble features
feature_cols = [
    "person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate",
    "loan_percent_income", "cb_person_cred_hist_length",
    "person_home_ownership_index", "loan_intent_index", "loan_grade_index", "cb_person_default_on_file_index"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Convert label to integer
df = df.withColumn("label", col("loan_status").cast("integer"))

# Save preprocessed data
df.select("features", "label").write.mode("overwrite").parquet("s3a://ml-bucket/preprocessed_data/data_credit_preprocessed.parquet")

print("✅ Data preprocessing selesai.")

spark.stop()