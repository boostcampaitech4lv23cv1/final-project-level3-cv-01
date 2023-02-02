import os
import sys
import cv2
import time
import tempfile
import uuid
from pathlib import Path
from pytz import timezone
from datetime import datetime
import streamlit as st
from google.cloud import storage

sys.path.append(os.getcwd())
import av
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import streamlit as st
from FastAPI.utils import upload_video, download_video

# Basic App Scaffolding
st.title("HEY-I")
st.subheader("면접 영상을 녹화하세요")
st.write("❗ 카메라 접근 권한을 승인해주세요 ❗")
st.markdown("##### 선택한 시간이 지나거나 End Recording 버튼을 누르면 녹화가 종료됩니다.")

# Create Sidebar
st.sidebar.title("Settings")

## Get Video
# temp_file = tempfile.NamedTemporaryFile(delete=False)
# number = st.sidebar.number_input("분 입력", 1, 10)
# stframe = st.empty()

st.markdown("**질문** : 1분 자기 소개를 해주세요")

start_time = datetime.now(timezone("Asia/Seoul")).strftime("%y%m%d_%H%M%S")
if "prefix" not in st.session_state:
    st.session_state["prefix"] = start_time
    # st.session_state["prefix"] = str(uuid.uuid4())
prefix = st.session_state["prefix"]

if not os.path.exists(f"./{st.session_state.name}_{st.session_state.num}/{prefix}"):
    os.makedirs(f"./{st.session_state.name}_{st.session_state.num}/{prefix}")
in_file = f"./{st.session_state.name}_{st.session_state.num}/{prefix}/recording.flv"
st.session_state.video_dir = in_file


########################################################### WebRTC
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    return av.VideoFrame.from_ndarray(img, format="bgr24")


def in_recorder_factory() -> MediaRecorder:
    return MediaRecorder(
        in_file, format="flv"
    )  # HLS does not work. See https://github.com/aiortc/aiortc/issues/331


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
st.session_state.video_dir = (
    f"./{st.session_state.name}_{st.session_state.num}/{prefix}/recording.webm"
)
with st.spinner("✔ 확인됐습니다. 변환 중입니다..."):
    start = time.process_time()
    cap = cv2.VideoCapture(in_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"vp80")

    out = cv2.VideoWriter(
        st.session_state.video_dir,
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
    print(f"Convert Complete: {st.session_state.video_dir} on {end-start}")

if "video_dir" in st.session_state.keys():
    if os.path.exists(st.session_state.video_dir):
        video_file = open(st.session_state.video_dir, "rb")
        video_bytes = video_file.read()
        st.write("녹화된 영상을 확인하시겠습니까?")
        with st.expander("가장 최근 녹화된 영상입니다. 이 영상을 분석 할 지 결정해주세요"):
            st.video(video_bytes)
            # 분석할 영상 결정
            st.write("이 영상으로 분석을 진행할까요?")
            confirm = st.button("Comfirm")
            if confirm:
                st.write("분석할 영상이 확인 되었습니다. Result 에서 결과를 확인하세요.")
                st.session_state.confirm_video = st.session_state.video_dir
