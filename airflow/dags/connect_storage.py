from google.cloud import storage

def download_video(storage_path, download_path, bucket_name="heyi-storage"):
    """_summary_

    Args:
        storage_path (_type_): 다운로드할 파일의 Google cloud의 파일 경로
        download_path (_type_): 다운로드할 파일의 현재 서버에서의 저장 경로
        bucket_name (str, optional): 다운로드할 bucket 이름. Defaults to "heyi-storage".
    """
    if '\\' in storage_path:
        storage_path = storage_path.replace('\\', '/')
    if '\\' in download_path:
        download_path = download_path.replace('\\', '/')

    storage_client = storage.Client.from_service_account_json(
        "./hey-i-375802-d3dcfd2b25d1.json"
    )
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(storage_path)
    blob.download_to_filename(download_path)
        
storage_client = storage.Client.from_service_account_json(
        "/opt/ml/final-project-level3-cv-01/airflow/hey-i-375802-d3dcfd2b25d1.json"
    )
bucket = storage_client.bucket("heyi-storage")
# blob = bucket.blob("*.webm")
# blob.download_to_filename("download")
import os
#os.system('gcloud storage cp gs://heyi-storage/* .')
os.system('gsutil cp -r gs://heyi-storage/CHO_1234 .')