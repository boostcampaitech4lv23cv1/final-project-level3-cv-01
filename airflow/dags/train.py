import pendulum
from datetime import time, timedelta, datetime
from pytz import timezone
from glob import glob
import splitfolders
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

def select_recent_videos(**context):
    now_date = datetime.now(timezone("Asia/Seoul")).strftime("%y%m%d")
    every_dir_bydate = glob('/opt/ml/final-project-level3-cv-01/airflow/heyi-storage/*/*')
    recent_dir = []
    for ed in every_dir_bydate:
        if int(now_date) - int(ed.split('/')[-1][0:6]) < 1 :
            recent_dir.append(ed)
    recent_videos = []
    for rd in recent_dir:
        recent_videos.append(f"{rd}/recording.webm")
    context['task_instance'].xcom_push(key='xcom_push_recent_videos', value=recent_videos)
    context['task_instance'].xcom_push(key='xcom_push_recent_dir', value=recent_dir)
    return recent_videos,recent_dir

def video_to_frame(**context):
    recent_videos = context['ti'].xcom_pull(key='xcom_push_recent_videos')
    saved_dirs = [f"{rv[:-15]}/frames" for rv in recent_videos]
    
    for rv,sd in zip(recent_videos,saved_dirs):
        
        if not os.path.exists(sd):
            os.makedirs(sd)

        cap = cv2.VideoCapture(rv)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS) / 20

        while True:  # 무한 루프
            ret, frame = cap.read()  # 두 개의 값을 반환하므로 두 변수 지정

            if not ret:  # 새로운 프레임을 못받아 왔을 때 braek
                break
            if int(cap.get(1)) % int(fps) == 0:
                cv2.imwrite(sd + "/frame%06d.jpg" % count, frame)
                print("Saved frame number : ", str(int(cap.get(1))))
                count += 1

            # 10ms 기다리고 다음 프레임으로 전환, Esc누르면 while 강제 종료
            if cv2.waitKey(10) == 27:
                break

        cap.release()  # 사용한 자원 해제
        cv2.destroyAllWindows()

def send_frame_to_dir(**context):
    recent_dir = context['ti'].xcom_pull(key='xcom_push_recent_dir')
    for sd in recent_dir:
        facedb = FaceDB(path=f"./{sd.split('/')[-2]}/{sd.split('/')[-1]}")
        df = facedb.load_data_train()
        for i in range(len(df)):
            source = f"/opt/ml/final-project-level3-cv-01/airflow/heyi-storage/{df.iloc[i]['frame'].lstrip('./')}"
            destination = f"face_dataset/train/{df.iloc[i]['emotion']}"
            print(f"{source} to {destination}")
            shutil.copy(source, destination)
            os.rename(f"{destination}/{df.iloc[i]['frame'].lstrip('./').split('/')[-1]}",f"{destination}/{sd.split('/')[-2]}_{sd.split('/')[-1]}_{df.iloc[i]['frame'].lstrip('./').split('/')[-1]}")

def split_valid():
    splitfolders.ratio('/opt/ml/final-project-level3-cv-01/airflow/face_dataset/train', 
                       output= '/opt/ml/final-project-level3-cv-01/airflow/face_dataset_train_valid' ,seed=42,ratio = (0.8,0.2))

def train_face():
    model = LightningModel.load_from_checkpoint('/opt/ml/final-project-level3-cv-01/model/face/models/custom_fer_model.ckpt')
    trainer = Trainer(
        max_epochs=10,        # val_check_interval = 1,
        accelerator="gpu",
        logger=CSVLogger(save_dir="./logs/"),
        callbacks=[

            ModelCheckpoint(
                dirpath = '/opt/ml/final-project-level3-cv-01/model/face/models/',
                filename="best_val_acc",
                verbose=True,
                save_last=True,
                save_top_k=1,
                monitor="val_acc",
                mode="max",
            ),
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=10),
        ],
    )

    trainer.fit(model)


with DAG(
    default_args=default_args,
    dag_id = "heyi_train",
    description = "train model by user data",
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
    
    t2 = PythonOperator(
        task_id= "select_data",
        python_callable =select_recent_videos,
        owner='jun',
        retries = 3,
        retry_delay = timedelta(minutes=3)
    )
    t3 = PythonOperator(
        task_id= "make_video_frame",
        python_callable =video_to_frame,
        owner='jun',
        retries = 3,
        retry_delay = timedelta(minutes=3)
    )
    
    t4= PythonOperator(
        task_id= "send_frame_to_dir",
        python_callable =send_frame_to_dir,
        owner='jun',
        retries = 3,
        retry_delay = timedelta(minutes=3)
    )
    t5 = PythonOperator(
        task_id= "train_face",
        python_callable =train_face,
        owner='jun',
        retries = 3,
        retry_delay = timedelta(minutes=3)
    )
t1 >> t2 >> t3 >> t4 >> t5
