import joblib
from io import BytesIO
from datetime import datetime
from airflow.models import Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook

def minio_do(method, key, bucket_name, data = None):
    s3 = S3Hook(aws_conn_id = 'minio-conn')

    if(method == "push"):
        joblib_buffer = BytesIO()
        joblib.dump(data, joblib_buffer)
        joblib_buffer.seek(0)
        
        s3.load_bytes(
            bytes_data = joblib_buffer.getvalue(),
            key = key,
            bucket_name = bucket_name,
            replace = True
        )

    elif(method == "pull"):
        pickle_object = s3.get_key(
            key = key,
            bucket_name = bucket_name
        )
        pickle_object = BytesIO(pickle_object.get()['Body'].read())
        pickle_object = joblib.load(pickle_object)

        return pickle_object
    
    else:
        raise RuntimeError(f"The parameter 'method' expected 'push' or 'pull', but {str(method)} is given.")

def xcom_do(ti, method, data = None, key = None, task_ids = None, include_prior_dates = False):
    if(method == "push"):
        if key == None:
            ti.xcom_push(
                key = ti.task_id,
                value = data
            )
        else:
            ti.xcom_push(
                key = key,
                value = data
            )    
    elif(method == "pull"):
        if(task_ids == None):
            raise RuntimeError(f"Task IDs not expected. {task_ids}.")
        
        else:
            return ti.xcom_pull(task_ids = task_ids, key = key, include_prior_dates = include_prior_dates)
        
    else:
        raise RuntimeError(f"The parameter 'method' expected 'push' or 'pull', but {str(method)} is given.")
    
def parse_datetime(date_string):
    return datetime.strptime(date_string, "%Y-%m-%d")

def variable_do(method, key, data = None, default = None):
    if(method == "get"):
        return Variable.get(key, default_var = default)
    
    elif(method == "set"):
        Variable.set(key, data)

        res = Variable.get(key, default_var = None)
        if res != data:
            raise RuntimeError(f"Variabel '{key}' has been tried to set to '{data}', but after verification got data '{res}'.")
    
def connect_database(db_conn_id):
    postgres_hook = PostgresHook(postgres_conn_id = db_conn_id)
    
    connection = postgres_hook.get_conn()
    cursor = connection.cursor()

    return connection, cursor