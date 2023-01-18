import streamlit as st
import sys
import os
import requests
import pandas as pd

BACKEND_FACE = 'http://127.0.0.1:8000/face_emotion'
BACKEND_POSE = 'http://127.0.0.1:8000/pose_estimation'

st.title('HEY-I')

if 'confirm_video' in st.session_state.keys():
    if os.path.exists(st.session_state.confirm_video):
        st.subheader('면접 영상 분석 결과입니다.')

        with st.expander('선택된 면접 영상입니다.'):
            video_file = open(st.session_state.confirm_video, 'rb')
            video_bytes = video_file.read()
            st.write('선택된 영상입니다.')
            st.video(video_bytes)
        
        inference = st.button('Inference')
        if inference:
            VIDEO_PATH = st.session_state.confirm_video
            SAVED_DIR = os.path.join(os.path.splitext(st.session_state.confirm_video)[0], 'frames')

            input_json = {
                'VIDEO_PATH' : VIDEO_PATH,
                'SAVED_DIR' : SAVED_DIR
            }


            with st.spinner('inferencing...'):
                r = requests.post(
                    BACKEND_FACE, json=input_json
                )
                r3 = requests.post(BACKEND_POSE, json=input_json)
            result = pd.read_json(r.text, orient = 'records')
            st.dataframe(result)
            st.write(r3.text)
    else:
        st.subheader('면접 영상이 제대로 저장되지 않았습니다. 다시 면접 영상을 녹화해주세요.')
else:
    st.subheader('면접 영상이 선택되지 않았습니다. 다시 면접 영상을 녹화해주세요.')




