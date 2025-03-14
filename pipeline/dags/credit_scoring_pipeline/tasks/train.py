from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier

# Initialize Spark Session
spark = SparkSession.builder.appName("TrainModel").getOrCreate()

# Load training data
train_data = spark.read.parquet("s3a://ml-bucket/preprocessed_data/data_credit_train.parquet")

# Train model
rf = RandomForestClassifier(
    featuresCol="features", 
    labelCol="label", 
    numTrees=50
)
model = rf.fit(train_data)

# Save trained model
model.write().overwrite().save("s3a://ml-bucket/model/ml_model_new")

print("Model training selesai.")

spark.stop()