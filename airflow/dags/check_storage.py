import pendulum
import os
from datetime import time, timedelta, datetime
from pytz import timezone
from glob import glob
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, task

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['quasar0529@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def save_webm(**context):
    webms = glob('/opt/ml/final-project-level3-cv-01/airflow/heyi-storage/*/*/*.webm')
    context['task_instance'].xcom_push(key='xcom_push_value', value=webms)
    return webms
def print_webm(**context):
    xcom_push_value = context['ti'].xcom_pull(key='xcom_push_value')
    # xcom_push_return_value = context['ti'].xcom_pull(task_ids='print_webm')
    print(xcom_push_value)
with DAG(
    default_args=default_args,
    dag_id = "heyi_check_storage",
    description = "check user data",
    schedule_interval =  " */5 * * * *", 
    start_date=days_ago(2),
    tags= ["heyi"],
) as dag:
    t1 = BashOperator(
        task_id = "print_dir",
        bash_command= "gsutil ls gs://heyi-storage/*/",
        owner= "jun",
        retries = 3,
        retry_delay = timedelta(minutes=3)
    )
    
    t2 = PythonOperator(
        task_id= "save_webm",
        python_callable =save_webm,
        owner='jun',
        retries = 3,
        retry_delay = timedelta(minutes=3)
    )
    
    t3 = PythonOperator(
        task_id= "print_webm",
        python_callable =print_webm,
        owner='jun',
        retries = 3,
        retry_delay = timedelta(minutes=3)
    )
    
t1 >> t2 >> t3