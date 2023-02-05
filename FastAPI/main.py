import os
import cv2
import sys
import glob
import json
import pandas as pd

sys.path.append(os.getcwd())

from copy import deepcopy
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

import model.face.utils.face_recognition_deepface as fr
from model.face.inference_pl import inference
from model.pose import pose_with_mediapipe as pwm
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
    VIDEO_PATH = download_path = inp.VIDEO_PATH
    # storage_path = os.path.join(*download_path.split("/")[1:])
    SAVED_DIR = inp.SAVED_DIR

    # if not os.path.exists(download_path):
    #     os.makedirs(os.path.join(*download_path.split("/")[1:-1]), exist_ok=True)
    #     download_video(storage_path=storage_path, download_path=download_path)
    #     print(f"The video was uploaded from {download_path} to {storage_path}")

    frames_dir = fr.video_to_frame(VIDEO_PATH, SAVED_DIR)
    print("frame_dir:", frames_dir)


@app.post("/face_emotion")
def get_emotion_df(inp: InferenceFace):
    VIDEO_PATH = download_path = inp.VIDEO_PATH
    storage_path = os.path.join(*download_path.split("/")[1:])
    SAVED_DIR = inp.SAVED_DIR

    # if not os.path.exists(download_path):
    #     os.makedirs(os.path.join(*download_path.split("/")[1:-1]), exist_ok=True)
    #     download_video(storage_path=storage_path, download_path=download_path)
    #     print(f"The video was uploaded from {download_path} to {storage_path}")

    # frames_dir = fr.video_to_frame(VIDEO_PATH, SAVED_DIR)
    # print("frame_dir:", frames_dir)

    output_dict, output_df = inference(
        32, "./model/face/models/custom_fer_model.ckpt", SAVED_DIR
    )
    output_df.sort_values(by=["frame"], ignore_index=True, inplace=True)
    # output_df.to_csv('./최명헌_5126/230204_021257/result.csv')

    rec_image_list = fr.add_emotion_on_frame_new(output_df)
    saved_video = fr.frame_to_video(rec_image_list, VIDEO_PATH)

    # frame_idx = face_analyze(df = output_df, threshold_sec = 0.4)
    # make_video_slice(df=output_df, frame_idx=frame_idx, video_path=saved_video, type='face')

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
    pose_df.to_csv('./최명헌_5126/230204_021257/pose_result.csv')

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

    # if not os.path.exists(download_path):
    #     os.makedirs(os.path.join(*download_path.split("/")[1:-1]), exist_ok=True)
    #     download_video(storage_path=storage_path, download_path=download_path)
    #     print(f"The video was uploaded from {download_path} to {storage_path}")

    # frames_dir = fr.video_to_frame(VIDEO_PATH, SAVED_DIR)
    # print("frame_dir:", frames_dir)

    frames = glob.glob(f"{SAVED_DIR}/*.jpg")
    frames.sort()
    df, anno_frames = gaze.analyze_eye(frames)
    df.to_csv('./최명헌_5126/230204_021257/eye_result.csv')

    saved_video = gaze.frame_to_video(VIDEO_PATH, anno_frames)

    uploaded_video = os.path.join(*saved_video.split("/")[1:])
    upload_video(saved_video, uploaded_video)

    df_json = df.to_json(orient="records")
    df_response = JSONResponse(json.loads(df_json))
    return df_response



def face_analyze(df, threshold_sec = 0.4):
    count = 0
    lst_all = []
    lst = []
    threshold = 20 * threshold_sec
    for idx, i in enumerate(df.posneg):
        # print(i)
        if i == 'negative':
            count += 1
            lst.append(idx)
        else:
            if count >= threshold:
                lst_all.append(deepcopy(lst))
            count = 0
            lst = []
    
    frame_idx = []
    
    if len(lst_all) > 0:
        for seq in lst_all:
            start = seq[0]
            end = seq[-1]
            frame_idx.append([start, end])
    else:
        pass

    return frame_idx

def make_video_slice(df, frame_idx, video_path, type):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # fps = mmcv.VideoReader(VIDEO_PATH).fps

    if not os.path.exists("./{video_path.split('/')[1]}/{video_path.split('/')[2]}/slice"):
        os.makedirs("./{video_path.split('/')[1]}/{video_path.split('/')[2]}/slice")

    slice_video_list = []

    for idx, frame in enumerate(frame_idx):

        start, end = frame
        fourcc = cv2.VideoWriter_fourcc(*"vp80")
        vid_save_name = f"./{video_path.split('/')[1]}/{video_path.split('/')[2]}/slice/{type}_slice_{idx}.webm"
        out = cv2.VideoWriter(vid_save_name, fourcc, fps, (width, height))

        for rec_frame in df['frame'][start, end+1]:
            out.write(rec_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        slice_video_list.append(vid_save_name)
        out.release()

    cap.release()
    cv2.destroyAllWindows()

    return slice_video_list



# if __name__ == '__main__':
#    uvicorn.run('FastAPI.main:app', host='0.0.0.0', port=8000, reload=True)
