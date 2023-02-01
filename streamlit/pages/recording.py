import os
import sys

sys.path.append(os.getcwd())

import cv2
import time
import tempfile
from pytz import timezone
from datetime import datetime
import requests
import pandas as pd
import streamlit as st
from google.cloud import storage
from FastAPI.utils import upload_video, download_video

if not 'name' in st.session_state.keys():
    st.warning('HEY-I 페이지에서 이름과 번호를 입력하세요')
    st.stop()

# BACKEND_POSE_MMPOSE = "http://49.50.175.182:30001/pose_with_mmpose"
BACKEND_FACE = "http://49.50.175.182:30001/face_emotion"
BACKEND_POSE_MMPOSE = "http://49.50.175.182:30001/pose_with_mmpose"
BACKEND_EYE = "http://49.50.175.182:30001/eye_tracking"
SAVE_REQUEST_DIR = "http://49.50.175.182:30001/save_origin_video"
UPLOAD_REQUEST_DIR = "http://49.50.175.182:30001/upload_predict_video"
# BACKEND_EYE = "http://127.0.0.1:8000/eye_tracking"
# SAVE_REQUEST_DIR = "http://127.0.0.1:8000/save_origin_video"
# UPLOAD_REQUEST_DIR = "http://127.0.0.1:8000/upload_predict_video"

# Basic App Scaffolding
st.title("HEY-I")
st.subheader("면접 영상을 녹화하세요")
st.markdown("##### 선택한 시간이 지나거나 End Recording 버튼을 누르면 녹화가 종료됩니다.")

# Create Sidebar
st.sidebar.title("Settings")

## Get Video
temp_file = tempfile.NamedTemporaryFile(delete=False)

number = st.sidebar.number_input("분 입력", 1, 10)
start_recording = st.sidebar.button("Start Recording")

if start_recording:
    st.markdown("**질문** : 1분 자기 소개를 해주세요")
    stframe = st.empty()
    with st.spinner("Get Ready for Camera"):
        video = cv2.VideoCapture(0)
        # Load Web Camera
        if not (video.isOpened()):
            print("File isn't opened!!")

        # Set Video File Property
        w = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        framecount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # video.set(cv2.CAP_PROP_FPS, 10) # fps 설정
        fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"vp80")
        # delay = 6
        print("fps:", fps)
        print("framecount:", framecount)

        # Save Video
        start_time = datetime.now(timezone("Asia/Seoul")).strftime("%y%m%d_%H%M%S")
        # if not os.path.exists(f"./{st.session_state.name}_{st.session_state.num}"):
        #     os.makedirs(f"./{st.session_state.name}_{st.session_state.num}")
        if not os.path.exists(
            f"./{st.session_state.name}_{st.session_state.num}/{start_time}"
        ):
            os.makedirs(
                f"./{st.session_state.name}_{st.session_state.num}/{start_time}"
            )

        video_dir = f"./{st.session_state.name}_{st.session_state.num}/{start_time}/recording.webm"
        st.session_state.video_dir = video_dir
        out = cv2.VideoWriter(video_dir, fourcc, fps/3, (w, h))
        if not (out.isOpened()):
            print("File isn't opened!!")
            video.release()
            sys.exit()

    end_recording = st.sidebar.button("End Recording")

    # Load frame and Save it
    start = time.time()
    timer = st.sidebar.empty()
    num_frames = 0
    while video.isOpened() and start_recording and not end_recording:
        ret, frame = video.read()

        sec = round(time.time() - start)
        timer.metric("Countdown", f"{sec//60:02d}:{sec%60:02d}")

        if ret and sec // 60 < number:
            num_frames += 1

            stframe.image(frame, channels="BGR", use_column_width=True)

            if start_recording:
                out.write(frame)

            cv2.waitKey(1)

        else:
            print("ret is false")
            break
    print("num frames:", num_frames)
    print()

    video.release()
    out.release()

    cv2.destroyAllWindows()


if "video_dir" in st.session_state.keys():
    if os.path.exists(st.session_state.video_dir):
        video_file = open(st.session_state.video_dir, "rb")
        video_bytes = video_file.read()
        st.write("가장 최근 녹화된 영상을 확인하시겠습니까?")
        check = st.checkbox("Check Video")
        if check:
            with st.expander("가장 최근 녹화된 영상입니다. 이 영상으로 업로드 할 것인지 결정해주세요"):
                st.video(video_bytes)

                # 분석할 영상 결정
                st.write("이 영상으로 분석을 진행할까요?")
                confirm = st.button("Inference")
                if confirm:
                    st.write("분석할 영상이 확인 되었습니다. Result 에서 결과를 확인하세요.")
                    with st.spinner('선택하신 영상을 분석하고 있습니다. 잠시 기다려주세요!'):
                        st.session_state.confirm_video = st.session_state.video_dir

                        # 녹화한 영상 cloud에 업로드할 경로
                        upload_path = os.path.join(
                            *st.session_state.video_dir.split("/")[-3:]
                        )
                        st.session_state.upload_dir = upload_path
                        upload_path = upload_path.replace('\\','/')

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

                        r_ = []
                        r_pose_=[]
                        with ThreadPoolExecutor () as executor:
                            r = executor.submit(requests.post, BACKEND_FACE, json=input_json)
                            r_pose = executor.submit(requests.post, BACKEND_POSE_MMPOSE, json=input_json)
                            r_.append(r)
                            r_pose_.append(r_pose)
                        # r = requests.post(BACKEND_FACE, json=input_json)
                        # r_pose = requests.post(BACKEND_POSE_MMPOSE, json=input_json)
                        # r_eye = requests.post(BACKEND_EYE, json=input_json)

                        result_dir = st.session_state.result_dir = os.path.join(*SAVED_DIR.split('/')[:-1])
                        for i in as_completed(r_):
                            r_result = i.result().text
                        for i in as_completed(r_pose_):
                            r_pose_result = i.result().text
                        result = pd.read_json(r_result, orient="records")
                        result.to_csv(os.path.join(result_dir, 'result.csv'))
                        pose_result = pd.read_json(r_pose_result, orient="records")
                        pose_result.to_csv(os.path.join(result_dir, 'pose_result.csv'))
                        # hand_result = pd.read_json(r_hand.json(), orient="records")

                        # Back에서 저장한 모델 예측 영상 경로 만들기
                        # for task in ("face", "pose", "eye"):
                        for task in ["face", "pose"]:
                            upload_name = task + "_" + st.session_state.upload_dir.split("\\")[-1]
                            upload_folder = os.path.join(
                                *st.session_state.upload_dir.split("\\")[:-1]
                            )
                            upload_dir = os.path.join(upload_folder, upload_name)
                            download_name = upload_name
                            download_folder = os.path.join(
                                *st.session_state.video_dir.split("/")[:-1]
                            )
                            download_dir = os.path.join(download_folder, download_name)

                            # 4. 클라우드에 저장된 모델 예측 영상 Front에 다운 받기
                            download_video(storage_path=upload_dir.replace('\\', '/'), download_path=download_dir)
                        st.success('분석이 완료 되었습니다!!!', icon = '🔥')
