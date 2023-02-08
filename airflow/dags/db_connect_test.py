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

#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from DBconnect.main import *
import shutil
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
import torch
from model.face.fer_pl import *
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['quasar0529@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}
def send_frame_to_dir(**context):
    recent_dir = context['ti'].xcom_pull(key='xcom_push_recent_dir')
    for sd in recent_dir:
        facedb = FaceDB(path=f"./{sd.split('/')[-2]/sd.split('/')[-1]}")
        df = facedb.load_data_train()
        for i in range(len(df)):
            source = f"/opt/ml/final-project-level3-cv-01/airflow/heyi-storage/{df.iloc[i]['frame'].lstrip('./')}"
            destination = f"face_dataset/{df.iloc[i]['emotion']}"
            print(f"{source} to {destination}")
            shutil.copy(source, destination)

with DAG(
    default_args=default_args,
    dag_id = "heyi_db",
    description = "connect db",
    schedule_interval = "0 0 * * MON", # 월요일 AM 12:00 마다 실행
    start_date = days_ago(2),
    tags= ["heyi"],
) as dag:
    t1 = BashOperator(
        task_id = "download_data",
        bash_command= "gcloud storage cp -r gs://heyi-storage/* /opt/ml/final-project-level3-cv-01/airflow/heyi-storage",
        owner= "jun",
        retries = 3,
        retry_delay = timedelta(minutes=3)
    )