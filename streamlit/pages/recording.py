import os
import sys
import cv2
import time
import shutil
from pathlib import Path
from pytz import timezone
from datetime import datetime
import streamlit as st
import io

sys.path.append(os.getcwd())
import av
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import requests
import pandas as pd
import streamlit as st
from google.cloud import storage
from FastAPI.utils import upload_video, download_video
from DBconnect.main import UserDB, PoseDB, EyeDB, FaceDB
if not "name" in st.session_state.keys():
    st.warning("HEY-I 페이지에서 이름과 번호를 입력하세요")
    st.stop()

assert os.path.exists("./hey-i-375802-d3dcfd2b25d1.json"), "Key가 존재하지 않습니다."


########################################################### WebRTC
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    return av.VideoFrame.from_ndarray(img, format="bgr24")


def convert_to_webm(in_file, video_dir):
    start = time.process_time()
    cap = cv2.VideoCapture(in_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"vp80")

    out = cv2.VideoWriter(
        video_dir,
        fourcc,
        fps,
        (width, height),
    )
    while True:
        ret, frame = cap.read()
        if not ret:  # 새로운 프레임을 못받아 왔을 때 braek
            break
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    end = time.process_time()
    
    print(f"Convert Complete: {video_dir} on {end-start}")


# BACKEND_FACE = "http://49.50.175.182:30001/face_emotion"
# BACKEND_POSE_MMPOSE = "http://49.50.175.182:30001/pose_with_mmpose"
# BACKEND_EYE = "http://49.50.175.182:30001/eye_tracking"
# SAVE_REQUEST_DIR = "http://49.50.175.182:30001/save_origin_video"
# UPLOAD_REQUEST_DIR = "http://49.50.175.182:30001/upload_predict_video"
BACKEND_FRAME = "http://127.0.0.1:8000/frames"
BACKEND_FACE = "http://127.0.0.1:8000/face_emotion"
BACKEND_POSE_MMPOSE = "http://127.0.0.1:8000/pose_with_mmpose"
BACKEND_EYE = "http://127.0.0.1:8000/eye_tracking"
SAVE_REQUEST_DIR = "http://127.0.0.1:8000/save_origin_video"
UPLOAD_REQUEST_DIR = "http://127.0.0.1:8000/upload_predict_video"

st.session_state.complete = False
st.session_state.cancel = False
st.session_state.recording = False

# Basic App Scaffolding
st.title("HEY-I")
st.subheader("면접 영상을 녹화하세요")

        
start_time = datetime.now(timezone("Asia/Seoul")).strftime("%y%m%d_%H%M%S")
if "prefix" not in st.session_state.keys() or st.session_state.prefix is None:
    st.session_state["prefix"] = start_time

if not os.path.exists(f"./{st.session_state.name}_{st.session_state.num}/{st.session_state.prefix}"):
    os.makedirs(f"./{st.session_state.name}_{st.session_state.num}/{st.session_state.prefix}")

flv_file = f"./{st.session_state.name}_{st.session_state.num}/{st.session_state.prefix}/recording.flv"
webm_file = f"./{st.session_state.name}_{st.session_state.num}/{st.session_state.prefix}/recording.webm"

uploaded_video = st.sidebar.file_uploader("영상 업로드", type=['mp4', 'flv'])
if uploaded_video:
    st.session_state.recording = True
    g = io.BytesIO(uploaded_video.read())
    ext = uploaded_video.type.split('/')[-1]
    uploaded_file = f"./{st.session_state.name}_{st.session_state.num}/{st.session_state.prefix}/recording."+ext
    with open(uploaded_file, 'wb') as out:
        out.write(g.read())

    convert = st.button('영상이 업로드 되었습니다. 이 버튼을 눌러 변환하세요.')
    if convert:
        with st.spinner("✔ 변환 중입니다..."):
            convert_to_webm(uploaded_file, webm_file)
            st.session_state.video_dir = webm_file


def in_recorder_factory():
    return MediaRecorder(
        flv_file, format="flv"
    )  # HLS does not work. See https://github.com/aiortc/aiortc/issues/331
userdb = UserDB(st.session_state.name, st.session_state.num, start_time, "/".join(flv_file.split('/')[:-1]))
posedb = PoseDB(st.session_state.name, st.session_state.num, start_time, "/".join(flv_file.split('/')[:-1]))
eyedb = EyeDB(st.session_state.name, st.session_state.num, start_time, "/".join(flv_file.split('/')[:-1]))
facedb = FaceDB(st.session_state.name, st.session_state.num, start_time, "/".join(flv_file.split('/')[:-1]))

if "userdb" not in st.session_state:
    st.session_state["userdb"] = userdb
    userdb.save_data()
if "posedb" not in st.session_state:
    st.session_state["posedb"] = posedb
if "eyedb" not in st.session_state:
    st.session_state["eyedb"] = eyedb
if "facedb" not in st.session_state:
    st.session_state["facedb"] = facedb

if not st.session_state.recording and not os.path.exists(webm_file):
    st.write("❗ 카메라 접근 권한을 승인해주세요")
    st.markdown("**질문** : 1분 자기 소개를 해주세요")
    webrtc_streamer(
        key="record",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        video_frame_callback=video_frame_callback,
        in_recorder_factory=in_recorder_factory,
    )
    ###########################################################
    
    convert = st.button('영상을 다 녹화한 후 이 버튼을 눌러 저장하세요.')
    if convert:
        with st.spinner("✔ 확인됐습니다. 변환 중입니다..."):
            convert_to_webm(flv_file, webm_file)
            st.session_state.video_dir = webm_file

if "video_dir" in st.session_state.keys() and st.session_state.video_dir == webm_file:
    if os.path.exists(st.session_state.video_dir):
        video_file = open(st.session_state.video_dir, "rb")
        video_bytes = video_file.read()
        with st.expander("이 영상을 분석 할 지 결정해주세요"):
            st.video(video_bytes)
            # 분석할 영상 결정

        st.write("이 영상으로 분석을 진행할까요?")

        confirm = st.button("Inference")
        cancel = st.button("Re-Recording")

        if confirm:
            with st.spinner('선택한 영상을 분석하고 있습니다. 잠시 기다려주세요!'):
                st.session_state.confirm_video = st.session_state.video_dir

                # 녹화한 영상 cloud에 업로드할 경로
                upload_path = "/".join(st.session_state.video_dir.split("/")[-3:])
                st.session_state.upload_dir = upload_path
                upload_path = upload_path.replace("\\", "/")

                start = time.time()  # 업로드 시간 측정
                # 1. Front에서 녹화한 영상 클라우드에 업로드
                upload_video(
                    file_path=st.session_state.video_dir, upload_path=upload_path
                )
                print(f"Front에서 클라우드로 업로드한 영상 경로 {upload_path}")

                # Front 에서 저장한 영상 경로와 저장할 클라우드 경로
                save_input_json = {
                    "VIDEO_PATH": st.session_state.upload_dir,
                    "SAVED_DIR": st.session_state.video_dir,
                }
                # 2. 클라우드에 저장된 영상 Back에 다운
                temp = requests.post(SAVE_REQUEST_DIR, json=save_input_json)

                VIDEO_PATH = st.session_state.confirm_video
                SAVED_DIR = (
                    f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/frames"
                )
                print(VIDEO_PATH, SAVED_DIR)
                input_json = {"VIDEO_PATH": VIDEO_PATH, "SAVED_DIR": SAVED_DIR}
                from concurrent.futures import ThreadPoolExecutor, as_completed
                
                requests.post(BACKEND_FRAME, json=input_json)

                r_ = []
                r_pose_ = []
                r_eye_ = []
                with ThreadPoolExecutor() as executor:
                    r = executor.submit(requests.post, BACKEND_FACE, json=input_json)
                    r_.append(r)
                    r_pose = executor.submit(
                        requests.post, BACKEND_POSE_MMPOSE, json=input_json
                    )
                    r_pose_.append(r_pose)
                    r_eye = executor.submit(requests.post, BACKEND_EYE, json=input_json)
                    r_eye_.append(r_eye)

                result_dir = "/".join(SAVED_DIR.split("/")[:-1])
                st.session_state.result_dir = result_dir 

                for i in as_completed(r_):
                    r_result = i.result().text
                for i in as_completed(r_pose_):
                    r_pose_result = i.result().text
                for i in as_completed(r_eye_):
                    r_eye_result = i.result().text
                    
                facedb.save_data(r_result)
                posedb.save_data(r_pose_result)
                eyedb.save_data(r_eye_result)

                # Back에서 저장한 모델 예측 영상 경로 만들기
                # for task in ("face", "pose", "eye"):
                for task in ["face", "pose", "eye"]:
                    upload_name = (task + "_" + st.session_state.upload_dir.split("/")[-1])
                    upload_folder = "/".join(st.session_state.upload_dir.split("/")[:-1])
                    upload_dir = "/".join([upload_folder, upload_name])
                    download_name = upload_name
                    download_folder = "/".join(st.session_state.video_dir.split("/")[:-1])
                    download_dir = "/".join([download_folder, download_name])

                    # 4. 클라우드에 저장된 모델 예측 영상 Front에 다운 받기
                    download_video(
                        storage_path=upload_dir,
                        download_path=download_dir,
                    )
            st.session_state.complete = True

        elif cancel:
            st.session_state.cancel = True
            st.session_state.prefix = None

if 'complete' in st.session_state.keys() and st.session_state.complete:
    st.success("분석이 완료 되었습니다!!! Result 페이지에서 결과를 확인하세요!!!", icon="🔥")
    st.session_state.complete = False

if 'cancel' in st.session_state.keys() and st.session_state.cancel:
    restart = st.button('다시 녹화하세요')
    st.session_state.cancel = False