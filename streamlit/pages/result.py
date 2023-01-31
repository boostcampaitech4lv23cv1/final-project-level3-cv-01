import os
import sys

sys.path.append(os.getcwd())

import time
import streamlit as st
import requests
import pandas as pd
from google.cloud import storage
from FastAPI.utils import upload_video, download_video


BACKEND_FACE = "http://127.0.0.1:8000/face_emotion"
BACKEND_POSE_SHOULDER = "http://127.0.0.1:8000/shoulder_pose_estimation"
BACKEND_POSE_HAND = "http://127.0.0.1:8000/hand_pose_estimation"
BACKEND_EYE = "http://127.0.0.1:8000/eye_tracking"
SAVE_REQUEST_DIR = "http://127.0.0.1:8000/save_origin_video"
UPLOAD_REQUEST_DIR = "http://127.0.0.1:8000/upload_predict_video"
st.set_page_config(layout="wide")
st.title("HEY-I")
# print (os.getcwd()) #현재 디렉토리의
# print (os.path.realpath(__file__))#파일

###
# key 파일 존재여부
print("isfile : ", os.path.isfile("hey-i-375802-e6e402d22694.json"))
if os.path.isfile("hey-i-375802-e6e402d22694.json"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "hey-i-375802-e6e402d22694.json"
    print("FINDING KEY SUCCEED!")
else:
    print("COULD NOT FIND KEY")


###
if "confirm_video" in st.session_state.keys():
    if os.path.exists(st.session_state.confirm_video):
        st.subheader("면접 영상 분석 결과입니다.")

        with st.expander("선택된 면접 영상입니다."):
            video_file = open(st.session_state.confirm_video, "rb")
            video_bytes = video_file.read()
            st.write("선택된 영상입니다.")
            st.video(video_bytes)

        inference = st.button("Inference")
        if inference:

            save_input_json = {
                "VIDEO_PATH": st.session_state.upload_dir,
                "SAVED_DIR": st.session_state.video_dir,
            }
            temp = requests.post(SAVE_REQUEST_DIR, json=save_input_json)

            VIDEO_PATH = st.session_state.confirm_video
            SAVED_DIR = (
                f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/frames"
            )
            input_json = {"VIDEO_PATH": VIDEO_PATH, "SAVED_DIR": SAVED_DIR}

            with st.spinner("inferencing..."):
                r = requests.post(BACKEND_FACE, json=input_json)
                r_shoulder = requests.post(BACKEND_POSE_SHOULDER, json=input_json)
                r_hand = requests.post(BACKEND_POSE_HAND, json=input_json)
                r_eye = requests.post(BACKEND_EYE, json=input_json)

            result = pd.read_json(r.text, orient="records")
            eye_result = pd.read_json(r_eye.text, orient="records")
            shoulder_result = pd.read_json(r_shoulder.json(), orient="records")
            hand_result = pd.read_json(r_hand.json(), orient="records")

            for task in ("face", "pose", "eye"):
                upload_name = task + "_" + st.session_state.upload_dir.split("/")[-1]
                upload_folder = os.path.join(
                    *st.session_state.upload_dir.split("/")[:-1]
                )
                upload_dir = os.path.join(upload_folder, upload_name)
                download_name = upload_name
                download_folder = os.path.join(
                    *st.session_state.video_dir.split("/")[:-1]
                )
                download_dir = os.path.join(download_folder, download_name)

                upload_input_json = {
                    "VIDEO_PATH": upload_dir,
                    "SAVED_DIR": download_dir,
                }
                temp = requests.post(UPLOAD_REQUEST_DIR, json=upload_input_json)

                download_video(storage_path=upload_dir, download_path=download_dir)

            tab1, tab2, tab3 = st.tabs(["Emotion", "Pose", "Eye"])

            with tab1:
                st.header("Emotion")
                st.subheader("니 얼굴 표정 이렇다 임마 표정 좀 풀어라")
                video_file = open(
                    f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/face_recording.webm",
                    "rb",
                )
                video_bytes = video_file.read()
                st.video(video_bytes)
                st.line_chart(result)

            with tab2:
                st.header("Pose")
                st.subheader("니 자세가 이렇다 삐딱하이 에픽하이")

                # pose estimation
                pose_video = open(
                    f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/pose_recording.webm",
                    "rb",
                )
                pose_video_bytes = pose_video.read()
                st.video(pose_video_bytes)

                shoulder_result = pd.read_json(r_shoulder.json(), orient="records")
                st.write("SHOULDER")
                st.dataframe(shoulder_result)

                hand_result = pd.read_json(r_hand.json(), orient="records")
                st.write("HAND")
                st.dataframe(hand_result)

            with tab3:
                st.header("Eye")
                st.subheader("동태눈깔 꼬라보노 보노보노")
                st.write("None : 정면 | Side: 그 외")
                st.dataframe(eye_result)
                video_file = open(
                    f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/eye_recording.webm",
                    "rb",
                )
                video_bytes = video_file.read()
                st.video(video_bytes)

    else:
        st.subheader("면접 영상이 제대로 저장되지 않았습니다. 다시 면접 영상을 녹화해주세요.")
else:
    st.subheader("면접 영상이 선택되지 않았습니다. 다시 면접 영상을 녹화해주세요.")
