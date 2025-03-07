from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier

# Initialize Spark Session
spark = SparkSession.builder.appName("TrainModel").getOrCreate()

# Load preprocessed data dari MinIO
df = spark.read.parquet("s3a://ml-bucket/preprocessed_data/data_credit_preprocessed.parquet")

# Split data (80% training, 20% testing)
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train model
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50)
model = rf.fit(train_data)

# Save trained model
model.write().overwrite().save("s3a://ml-bucket/model/ml_model_new")

print("✅ Model training selesai.")

spark.stop()