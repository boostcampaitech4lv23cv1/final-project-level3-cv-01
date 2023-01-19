import streamlit as st
import sys
import os
import requests
import pandas as pd
import time

BACKEND_FACE = "http://127.0.0.1:8000/face_emotion"

st.title("HEY-I")

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
            VIDEO_PATH = st.session_state.confirm_video
            SAVED_DIR = os.path.join(
                os.path.splitext(st.session_state.confirm_video)[0], "frames"
            )

            input_json = {"VIDEO_PATH": VIDEO_PATH, "SAVED_DIR": SAVED_DIR}

            # with st.spinner("inferencing..."):
            #     try:
            #         r = requests.post(BACKEND_FACE, json=input_json)
            #     except:
            #         time.sleep(2)
            #         r = requests.post(BACKEND_FACE, json=input_json)
            #     r = requests.post(BACKEND_FACE, json=input_json)
            with st.spinner("inferencing..."):
                r = requests.post(BACKEND_FACE, json=input_json)
            video_file = open("./db/vp80.webm", "rb")
            video_bytes = video_file.read()
            st.video(video_bytes)

            result = pd.read_json(r.text, orient="records")
            # st.dataframe(result)
            tab1, tab2, tab3 = st.tabs(["Emotion", "Pose", "Eye"])

            with tab1:
                st.header("Emotion")
                st.subheader("니 얼굴 표정 이렇다 임마 표정 좀 풀어라")
                st.line_chart(result)
            with tab2:
                st.header("Pose")
                st.subheader("니 자세가 이렇다 삐딱하이 에픽하이")

            with tab3:
                st.header("Eye")
                st.subheader("동태눈깔 꼬라보노 보노보노")

    else:
        st.subheader("면접 영상이 제대로 저장되지 않았습니다. 다시 면접 영상을 녹화해주세요.")
else:
    st.subheader("면접 영상이 선택되지 않았습니다. 다시 면접 영상을 녹화해주세요.")
