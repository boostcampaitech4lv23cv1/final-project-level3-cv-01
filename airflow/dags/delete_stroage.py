import pendulum
from datetime import time, timedelta, datetime
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

# with DAG(
#     default_args = default_args,
#     dag_id = "heyi_delete",
#     description = "delete user data once a month",
#     schedule_interval = "0 0 1 * *", # 매달 1일 12AM 마다 실행
#     start_date = datetime(2023,2,1),
#     tags= ["heyi"],
# ) as dag:
#     t1 = BashOperator(
#         task_id = 'delete_data',
#         bash_command = "gsutil rm gs://heyi-storage",
#         owner = "jun",
#         retries = 1,
#         retry_delay = timedelta(minutes=3)
#     )
    
# t1
    