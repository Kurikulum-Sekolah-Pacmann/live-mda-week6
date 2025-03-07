from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from datetime import datetime

# Initialize Spark Session
spark = SparkSession.builder.appName("EvaluateModel").getOrCreate()

# Load preprocessed data
df = spark.read.parquet("s3a://ml-bucket/preprocessed_data/data_credit_preprocessed.parquet")

# Load model baru
new_model = RandomForestClassificationModel.load("s3a://ml-bucket/model/ml_model_new")

# Split data (80% training, 20% testing)
_, test_data = df.randomSplit([0.8, 0.2], seed=42)

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

# Simpan hasil evaluasi ke PostgreSQL
evaluation_result = [(datetime.now().isoformat(), new_accuracy, old_accuracy)]
df_eval = spark.createDataFrame(evaluation_result, ["date", "new_accuracy", "old_accuracy"])
# df_eval.write.jdbc(url="jdbc:postgresql://sources:5432/postgres", table="model_evaluation", mode="append")
print(df_eval.show())

# Bandingkan model baru dan lama
if new_accuracy > old_accuracy:
    print(f"✅ Model baru lebih baik ({new_accuracy:.4f} > {old_accuracy:.4f}). Model baru dideploy.")
    new_model.write().overwrite().save("s3a://ml-bucket/model/ml_model")
else:
    print(f"⚠️ Model lama lebih baik ({old_accuracy:.4f} >= {new_accuracy:.4f}). Model lama tetap digunakan.")

spark.stop()