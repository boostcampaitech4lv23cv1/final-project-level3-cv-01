import os
import sys
sys.path.append(os.getcwd())

import cv2
import time
import tempfile
from pytz import timezone
from datetime import datetime

import streamlit as st
from FastAPI.utils import upload_video, download_video

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
        if not os.path.exists(f"./{st.session_state.name}_{st.session_state.num}/{start_time}"):
            os.makedirs(f"./{st.session_state.name}_{st.session_state.num}/{start_time}")    
        
        video_dir = f"./{st.session_state.name}_{st.session_state.num}/{start_time}/recording.webm"
        st.session_state.video_dir = video_dir
        out = cv2.VideoWriter(video_dir, fourcc, fps / 4, (w, h))
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
                confirm = st.button("Comfirm")
                if confirm:
                    st.write("분석할 영상이 확인 되었습니다. Result 에서 결과를 확인하세요.")
                    st.session_state.confirm_video = st.session_state.video_dir
                    
                    
                    upload_path = os.path.join(*st.session_state.video_dir.split("/")[-3:])
                    st.session_state.upload_dir = upload_path
                    
                    upload_video(file_path=st.session_state.video_dir, upload_path=upload_path)
                    print(f"The video has been uploaded to {upload_path}")

                    


                    

