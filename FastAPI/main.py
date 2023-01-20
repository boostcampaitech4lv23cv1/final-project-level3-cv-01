import os, sys, json, uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.append(os.getcwd())
import model.face.face_recognition_deepface as fr
from model.pose import pose_with_mediapipe as pwm
import model.eye.gaze_tracking.gaze_tracking as gt

app = FastAPI(title="HEY-I", description="This is a demo of HEY-I")

# video를 받아와야 함
# Bytes 파일로 받을지 str로 받을지 결정해야 함. 우선은 str로 받는 버전
class InferenceFace(BaseModel):
    VIDEO_PATH: str
    SAVED_DIR: str


@app.get("/")
def base():
    return {"hello": "world"}


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
    frames = fr.video_to_frame(VIDEO_PATH, SAVED_DIR)
    df, anno_frames = gaze.analyze_eye(frames)
    df_json = df.to_json(orient="records")
    df_response = JSONResponse(json.loads(df_json))

    gaze.frame_to_video(anno_frames)
    return df_response


# if __name__ == '__main__':
#     uvicorn.run('FastAPI.main:app', host='0.0.0.0', port=8000, reload=True)
