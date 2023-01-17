import os, sys, json, uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.append(os.getcwd())
import model.face.face_recognition_deepface as fr
import model.eye.gaze_tracking.gaze_tracking as gt

app = FastAPI(
    title='HEY-I',
    description='This is a demo of HEY-I'
)

# video를 받아와야 함
# Bytes 파일로 받을지 str로 받을지 결정해야 함. 우선은 str로 받는 버전
class InferenceFace(BaseModel):
    VIDEO_PATH: str
    SAVED_DIR: str

@app.get("/")
def base():
    return {"hello" : "world"}

@app.post("/face_emotion")
def get_emotion_df(inp: InferenceFace):
    VIDEO_PATH = inp.VIDEO_PATH
    SAVED_DIR = inp.SAVED_DIR
    frames = fr.video_to_frame(VIDEO_PATH, SAVED_DIR)
    emotions_mtcnn = fr.emotion(frames)
    df = fr.make_emotion_df(emotions_mtcnn)
    df_json = df.to_json(orient='records')
    df_response = JSONResponse(json.loads(df_json))
    return df_response


@app.post("/eye_tracking")
def get_eye_df(inp: InferenceFace):
    gaze = gt()
    VIDEO_PATH = inp.VIDEO_PATH
    SAVED_DIR = inp.SAVED_DIR
    frames = fr.video_to_frame(VIDEO_PATH, SAVED_DIR)
    df = gaze.analyze_eye(frames)
    df_json = df.to_json(orient='records')
    df_response = JSONResponse(json.loads(df_json))
    return df_response


# if __name__ == '__main__':
#     uvicorn.run('FastAPI.main:app', host='127.0.0.1', port=8000, reload=True)
