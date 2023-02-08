import pendulum
from datetime import time, timedelta, datetime
from pytz import timezone
from glob import glob
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, task
import os
import cv2
import sys
sys.path.append('/opt/ml/final-project-level3-cv-01')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['quasar0529@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}
def select_recent_videos(**context):
    now_date = datetime.now(timezone("Asia/Seoul")).strftime("%y%m%d")
    every_dir_bydate = glob('/opt/ml/final-project-level3-cv-01/airflow/heyi-storage/*/*')
    recent_dir = []
    for ed in every_dir_bydate:
        if int(now_date) - int(ed.split('/')[-1][0:6]) > 5 :
            recent_dir.append(ed)
    context['task_instance'].xcom_push(key='xcom_push_recent_dir', value=recent_dir)
    return recent_dir
# with DAG(
#     default_args = default_args,
#     dag_id = "heyi_delete",
#     description = "delete user data once a month",
#     schedule_interval = "0 0 1 * *", # 매달 1일 12AM 마다 실행
#     start_date = datetime(2023,2,2),
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
    