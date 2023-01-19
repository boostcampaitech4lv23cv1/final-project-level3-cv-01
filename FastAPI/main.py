import os, sys, json, uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd

sys.path.append(os.getcwd())
import model.face.face_recognition_deepface as fr

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

    pos_emo = ["happy", "neutral"]
    neg_emp = ["angry", "disgust", "fear", "sad", "surprise"]

    positive = []
    negative = []

    for i in range(1, len(emotions_mtcnn) + 1):
        tmp = "instance_" + str(i)
        p = 0
        n = 0
        if emotions_mtcnn[tmp]["dominant_emotion"] in pos_emo:
            p += emotions_mtcnn[tmp]["emotion"]["happy"]
            p += emotions_mtcnn[tmp]["emotion"]["neutral"]

        else:
            n += emotions_mtcnn[tmp]["emotion"]["angry"]
            n += emotions_mtcnn[tmp]["emotion"]["disgust"]
            n += emotions_mtcnn[tmp]["emotion"]["fear"]
            n += emotions_mtcnn[tmp]["emotion"]["sad"]
            n += emotions_mtcnn[tmp]["emotion"]["surprise"]
        positive.append(p)
        negative.append(n)
    df_binary = pd.DataFrame({"positive": positive, "negative": negative})

    df_json = df_binary.to_json(orient="records")
    df_response = JSONResponse(json.loads(df_json))
    return df_response


# if __name__ == '__main__':
#     uvicorn.run('FastAPI.main:app', host='127.0.0.1', port=8000, reload=True)
