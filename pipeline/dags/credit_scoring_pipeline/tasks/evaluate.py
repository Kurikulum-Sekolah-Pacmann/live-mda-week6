from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from datetime import datetime

# Initialize Spark Session
spark = SparkSession.builder.appName("EvaluateModel").getOrCreate()

# Load test_data
test_data = spark.read.parquet("s3a://ml-bucket/preprocessed_data/data_credit_test.parquet")

# Load new_models
new_model = RandomForestClassificationModel.load("s3a://ml-bucket/model/ml_model_new")

# Evaluasi model baru
new_predictions = new_model.transform(test_data)
new_accuracy = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy").evaluate(new_predictions)

# Load model lama (jika ada)
try:
    old_model = RandomForestClassificationModel.load("s3a://ml-bucket/model/ml_model")
    old_predictions = old_model.transform(test_data)
    old_accuracy = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy").evaluate(old_predictions)
except:
    old_accuracy = 0  # Jika model lama tidak ada

# Bandingkan model baru dan lama
if new_accuracy > old_accuracy:
    print(f"✅ Model baru lebih baik ({new_accuracy:.4f} > {old_accuracy:.4f}). Model baru dideploy.")
    new_model.write().overwrite().save("s3a://ml-bucket/model/ml_model")
else:
    print(f"⚠️ Model lama lebih baik ({old_accuracy:.4f} >= {new_accuracy:.4f}). Model lama tetap digunakan.")

spark.stop()