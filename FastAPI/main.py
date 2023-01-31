import os
import sys
import glob
import json #, uvicorn
sys.path.append(os.getcwd())

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

import model.face.face_recognition_deepface as fr
from model.pose import pose_with_mediapipe as pwm
import model.eye.gaze_tracking.gaze_tracking as gt
from FastAPI.utils import upload_video, download_video

app = FastAPI(title="HEY-I", description="This is a demo of HEY-I")

# video를 받아와야 함
# Bytes 파일로 받을지 str로 받을지 결정해야 함. 우선은 str로 받는 버전
class InferenceFace(BaseModel):
    VIDEO_PATH: str
    SAVED_DIR: str


@app.get("/")
def base():
    return {"hello": "world"}

@app.post("/save_origin_video")
#def save_video(storage_path:str, download_path:str):
def save_video(inp: InferenceFace):
    storage_path = inp.VIDEO_PATH
    download_path = inp.SAVED_DIR
    os.makedirs(os.path.join(*download_path.split("/")[1:-1]), exist_ok=True)
    download_video(storage_path=storage_path, download_path=download_path)
    return storage_path, download_path

@app.post("/upload_predict_video")
#def save_video(storage_path:str, download_path:str):
def save_video(inp: InferenceFace):
    storage_path = inp.VIDEO_PATH
    download_path = inp.SAVED_DIR
    print("!!!!!!storage_path",storage_path)
    print("!!!!!!download_path",download_path)
    #os.makedirs(os.path.join(*download_path.split("/")[1:-1]), exist_ok=True)
    #download_video(storage_path=storage_path, download_path=download_path)
    upload_video(file_path=download_path, upload_path=storage_path)
    return storage_path, download_path


@app.post("/face_emotion")
def get_emotion_df(inp: InferenceFace):
    VIDEO_PATH = inp.VIDEO_PATH
    SAVED_DIR = inp.SAVED_DIR
    frames = fr.video_to_frame(VIDEO_PATH, SAVED_DIR)
    emotions_mtcnn = fr.analyze_emotion(frames)
    df = fr.make_emotion_df(emotions_mtcnn)
    rec_image_list = fr.add_emotion_on_frame(emotions_mtcnn, df, SAVED_DIR)
    fr.frame_to_video(rec_image_list, VIDEO_PATH)

    df_binary = fr.make_binary_df(emotions_mtcnn, df)

    df_json = df_binary.to_json(orient="records")
    df_response = JSONResponse(json.loads(df_json))
    return df_response


@app.post("/shoulder_pose_estimation")
def get_shoulder_results(inp: InferenceFace):
    VIDEO_PATH = inp.VIDEO_PATH
    shoulder_info, _ = pwm.run(VIDEO_PATH)
    shoulder_json = pwm.dict_to_json(shoulder_info)
    shoulder_response = JSONResponse(shoulder_json)
    return shoulder_response


@app.post("/hand_pose_estimation")
def get_hand_results(inp: InferenceFace):
    VIDEO_PATH = inp.VIDEO_PATH
    _, hand_info = pwm.run(VIDEO_PATH)
    hand_json = pwm.dict_to_json(hand_info)
    hand_response = JSONResponse(hand_json)
    return hand_response


@app.post("/eye_tracking")
def get_eye_df(inp: InferenceFace):
    gaze = gt.GazeTracking()
    VIDEO_PATH = inp.VIDEO_PATH
    SAVED_DIR = inp.SAVED_DIR
    frames = glob.glob(f"{SAVED_DIR}/*.jpg")
    frames.sort()
    df, anno_frames = gaze.analyze_eye(frames)
    df_json = df.to_json(orient="records")
    df_response = JSONResponse(json.loads(df_json))

    gaze.frame_to_video(VIDEO_PATH,anno_frames)
    return df_response


# if __name__ == '__main__':
#     uvicorn.run('FastAPI.main:app', host='0.0.0.0', port=8000, reload=True)
