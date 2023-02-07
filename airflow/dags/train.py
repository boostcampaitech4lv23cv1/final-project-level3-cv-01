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
        if int(now_date) - int(ed.split('/')[-1][0:6]) < 2:
            recent_dir.append(ed)
    recent_videos = []
    for rd in recent_dir:
        recent_videos.append(f"{rd}/recording.webm")
    context['task_instance'].xcom_push(key='xcom_push_recent_videos', value=recent_videos)
    return recent_videos

def video_to_frame(**context):
    recent_videos = context['ti'].xcom_pull(key='xcom_push_recent_videos')
    saved_dirs = [f"{rv[:-15]}/frames" for rv in recent_videos]
    
    for rv,sd in zip(recent_videos,saved_dirs):
        
        if not os.path.exists(sd):
            os.makedirs(sd)

        cap = cv2.VideoCapture(rv)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:  # 무한 루프
            ret, frame = cap.read()  # 두 개의 값을 반환하므로 두 변수 지정

            if not ret:  # 새로운 프레임을 못받아 왔을 때 braek
                break
            if int(cap.get(1)) % int(fps / 3) == 0:
                cv2.imwrite(sd + "/frame%d.jpg" % count, frame)
                print("Saved frame number : ", str(int(cap.get(1))))
                count += 1

            # 10ms 기다리고 다음 프레임으로 전환, Esc누르면 while 강제 종료
            if cv2.waitKey(10) == 27:
                break

        cap.release()  # 사용한 자원 해제
        cv2.destroyAllWindows()

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
        bash_command= "gcloud storage cp -r gs://heyi-storage/* ..",
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
    
t1 >> t2 >> t3