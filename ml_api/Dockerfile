FROM python:3.12

# Install OpenJDK 17 karena OpenJDK 11 tidak tersedia
RUN apt-get update && apt-get install -y openjdk-17-jdk && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME agar PySpark bisa menemukannya
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Install dependencies
WORKDIR /app
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Download Hadoop AWS & AWS SDK Bundle
RUN mkdir -p /opt/spark/jars && \
    curl -o /opt/spark/jars/hadoop-aws-3.3.4.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    curl -o /opt/spark/jars/aws-java-sdk-bundle-1.12.262.jar https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar

# Set environment variable Spark Master
ENV SPARK_MASTER_URL=spark://spark-master:7077

# Copy FastAPI script
COPY fastapi_serving.py /app

# Expose port FastAPI
EXPOSE 8000

# Jalankan FastAPI
CMD ["uvicorn", "fastapi_serving:app", "--host", "0.0.0.0", "--port", "8000"]