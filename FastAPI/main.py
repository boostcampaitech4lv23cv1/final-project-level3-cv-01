import os
import sys
import glob
import json
import pandas as pd

sys.path.append(os.getcwd())

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

import model.face.utils.face_recognition_deepface as fr
from model.face.inference_pl import inference
import model.eye.gaze_tracking.gaze_tracking as gt
from model.pose.pose_with_mmpose import main
from FastAPI.utils import upload_video, download_video
from typing import List

app = FastAPI(title="HEY-I", description="This is a demo of HEY-I")

# video를 받아와야 함
# Bytes 파일로 받을지 str로 받을지 결정해야 함. 우선은 str로 받는 버전
class InferenceFace(BaseModel):
    VIDEO_PATH: str
    SAVED_DIR: str


class Item(BaseModel):
    frame_id: List[int]
    shoulder_angle: List[int]
    hands_on: List[int]


@app.get("/")
def base():
    return {"hello": "world"}


# Front에서 cloud에 저장된 원본 영상 Back에 저장
@app.post("/save_origin_video")
def save_origin_video(inp: InferenceFace):
    storage_path = inp.VIDEO_PATH
    download_path = inp.SAVED_DIR
    os.makedirs(os.path.join(*download_path.split("/")[1:-1]), exist_ok=True)
    download_video(storage_path=storage_path, download_path=download_path)
    print(f"The video was uploaded from {download_path} to {storage_path}")
    return storage_path, download_path


# Back에서 저장한 모델 예측 영상 cloud에 저장
@app.post("/upload_predict_video")
def upload_predict_video(inp: InferenceFace):
    storage_path = inp.VIDEO_PATH
    download_path = inp.SAVED_DIR
    upload_video(file_path=download_path, upload_path=storage_path)
    print(f"The video was saved from {storage_path} to {download_path}")
    return storage_path, download_path


@app.post("/frames")
def make_frame(inp: InferenceFace):
    VIDEO_PATH = inp.VIDEO_PATH
    SAVED_DIR = inp.SAVED_DIR

    frames_dir = fr.video_to_frame(VIDEO_PATH, SAVED_DIR)
    print("frame_dir:", frames_dir)


@app.post("/face_emotion")
def get_emotion_df(inp: InferenceFace):
    VIDEO_PATH = download_path = inp.VIDEO_PATH
    storage_path = os.path.join(*download_path.split("/")[1:])
    SAVED_DIR = inp.SAVED_DIR

    output_dict, output_df = inference(
        32, "./model/face/models/custom_fer_model.ckpt", SAVED_DIR
    )
    output_df.sort_values(by=["frame"], ignore_index=True, inplace=True)

    rec_image_list = fr.add_emotion_on_frame_new(output_df)
    saved_video = fr.frame_to_video(rec_image_list, VIDEO_PATH)

    uploaded_video = os.path.join(*saved_video.split("/")[1:])
    upload_video(saved_video, uploaded_video)

    df_json = output_df.to_json(orient="records")
    df_response = JSONResponse(json.loads(df_json))
    return df_response


@app.post("/pose_with_mmpose")
def demo_with_mmpose(inp: InferenceFace):
    VIDEO_PATH = download_path = inp.VIDEO_PATH
    storage_path = os.path.join(*download_path.split("/")[1:])
    SAVED_DIR = inp.SAVED_DIR

    if not os.path.exists(download_path):
        os.makedirs(os.path.join(*download_path.split("/")[1:-1]), exist_ok=True)
        download_video(storage_path=storage_path, download_path=download_path)
        print(f"The video was uploaded from {download_path} to {storage_path}")

    pose_dict = main(VIDEO_PATH, SAVED_DIR)
    pose_df = pd.DataFrame(pose_dict)
    saved_video = (
        "/".join(SAVED_DIR.split("/")[:-1]) + "/pose_" + os.path.basename(VIDEO_PATH)
    )
    uploaded_video = os.path.join(*saved_video.split("/")[1:])
    upload_video(saved_video, uploaded_video)

    pose_json = pose_df.to_json(orient="records")
    pose_response = JSONResponse(json.loads(pose_json))
    return pose_response


@app.post("/eye_tracking")
def get_eye_df(inp: InferenceFace):
    gaze = gt.GazeTracking()
    VIDEO_PATH = download_path = inp.VIDEO_PATH
    storage_path = os.path.join(*download_path.split("/")[1:])
    SAVED_DIR = inp.SAVED_DIR

    frames = glob.glob(f"{SAVED_DIR}/*.jpg")
    frames.sort()
    df, anno_frames = gaze.analyze_eye(frames)

    saved_video = gaze.frame_to_video(VIDEO_PATH, anno_frames)

    uploaded_video = os.path.join(*saved_video.split("/")[1:])
    upload_video(saved_video, uploaded_video)

    df_json = df.to_json(orient="records")
    df_response = JSONResponse(json.loads(df_json))
    return df_response
