from fastapi import FastAPI, HTTPException
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pydantic import BaseModel
import os

# Inisialisasi FastAPI
app = FastAPI()

# Konfigurasi Spark Master (sesuaikan dengan alamat Spark Anda)
SPARK_MASTER_URL = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")

# PYTHON_PATH = "/usr/bin/python3.9"

# Inisialisasi Spark Session
spark = SparkSession.builder \
    .appName("ModelServing") \
    .master(SPARK_MASTER_URL) \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "1g") \
    .config("spark.hadoop.fs.s3a.access.key", "minio") \
    .config("spark.hadoop.fs.s3a.secret.key", "minio123") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .getOrCreate()


# Load Model dari MinIO
MODEL_PATH = "s3a://ml-bucket/model/ml_model"
try:
    model = RandomForestClassificationModel.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    model = None

# Definisi Input Data Model
class ModelInput(BaseModel):
    person_age: float
    person_income: float
    person_emp_length: float
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    person_home_ownership_index: int
    loan_intent_index: int
    loan_grade_index: int
    cb_person_default_on_file_index: int

# Endpoint untuk melakukan prediksi
@app.post("/predict")
def predict(input_data: ModelInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")

    # Konversi input ke DataFrame Spark
    data_dict = input_data.dict()
    df = spark.createDataFrame([data_dict])

    # Assemble fitur
    feature_cols = [
        "person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate",
        "loan_percent_income", "cb_person_cred_hist_length",    
        "person_home_ownership_index", "loan_intent_index", "loan_grade_index", "cb_person_default_on_file_index"
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    # Melakukan prediksi
    predictions = model.transform(df)

    # Ambil hasil prediksi
    result = predictions.select("prediction").collect()[0]["prediction"]
    return {"prediction": int(result)}

# Menjalankan FastAPI jika script dieksekusi langsung
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
