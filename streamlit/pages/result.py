import os
import sys
import cv2
from copy import deepcopy

sys.path.append(os.getcwd())

import time
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from collections import defaultdict
from google.cloud import storage
from FastAPI.utils import upload_video, download_video

# 시간 측정

cls_to_idx = {
    "angry":0,
    "anxiety":1,
    "happy":2,
    "hurt":3,
    "neutral":4,
    "sad":5,
    "surprise":6,
}

elapsed_time = dict()

if not 'name' in st.session_state.keys():
    st.warning('HEY-I 페이지에서 이름과 번호를 입력하세요')
    st.stop()

BACKEND_FACE = "http://127.0.0.1:8000/face_emotion"
BACKEND_POSE_SHOULDER = "http://127.0.0.1:8000/shoulder_pose_estimation"
BACKEND_POSE_HAND = "http://127.0.0.1:8000/hand_pose_estimation"
BACKEND_EYE = "http://127.0.0.1:8000/eye_tracking"
SAVE_REQUEST_DIR = "http://127.0.0.1:8000/save_origin_video"
UPLOAD_REQUEST_DIR = "http://127.0.0.1:8000/upload_predict_video"

st.set_page_config(layout="wide")
st.title("HEY-I")

# key 존재 확인
assert os.path.exists("./hey-i-375802-e6e402d22694.json"), "Key가 존재하지 않습니다."

if 'result_dir' in st.session_state.keys():
    if os.path.exists(st.session_state.confirm_video):
        st.subheader("면접 영상 분석 결과입니다.")

        # check = st.checkbox('선택된 면접 영상을 확인하시겠습니까?')

        # if check:
        #     with st.expander("선택된 면접 영상입니다."):
        #         video_file = open(st.session_state.confirm_video, "rb")
        #         video_bytes = video_file.read()
        #         st.write("선택된 영상입니다.")
        #         st.video(video_bytes)

        VIDEO_PATH = st.session_state.confirm_video
        result = pd.read_csv(os.path.join(st.session_state.result_dir, 'result.csv'), index_col=0)
        tab1, tab2, tab3 = st.tabs(["Emotion", "Pose", "Eye"])

        with tab1:
            st.header("Emotion")
            st.subheader("니 얼굴 표정 이렇다 임마 표정 좀 풀어라")
            video = cv2.VideoCapture(f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/face_recording.webm")
            video_len = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
            sec = [video_len / len(result) * (i + 1) for i in range(len(result))]
            result['seconds'] = sec
            video_file = open(
                f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/face_recording.webm",
                "rb",
            )
            video_bytes = video_file.read()
            st.video(video_bytes)
            # st.line_chart([cls_to_idx[i] for i in result.emotion])
            
            linechart = st.selectbox(
                'What kind of line chart do you want?',
                ('Emotion (7 classes)', 'Positive or Negative', 'Both')
                )

            fig, ax = plt.subplots()
            ax.set_xlabel('Time(sec)')
            ax.set_ylabel('Emotion')
            ax.set_xticks([i+1 for i in range(len(result)) if i%5 == 0])
            ax.set_xticklabels([round(j, 1) for i, j in enumerate(result.seconds) if i%5==0])
            ax.tick_params(axis='x', rotation=30)

            if linechart == 'Emotion (7 classes)':
                ax.plot(result.emotion, color = 'skyblue', label = 'emotion')
                # ax.tick_params(bottom=False)
                ax.set_yticks(['neutral','happy','angry','anxiety','sad','surprise', 'hurt'])
                ax.set_yticklabels(['neutral','happy','angry','anxiety','sad','surprise', 'hurt'])
                st.pyplot(fig)

            elif linechart == 'Positive or Negative':
                ax.plot(result.posneg, color = 'salmon')
                ax.set_yticks(['positive','negative'])
                ax.set_yticklabels(['Positive', 'Negative'])
                st.pyplot(fig)
            
            elif linechart == 'Both':
                ax.plot(result.emotion, color = 'skyblue', label = 'emotion')
                # ax.tick_params(bottom=False)
                ax.set_yticks(['neutral','happy','angry','anxiety','sad','surprise', 'hurt'])
                ax.set_yticklabels(['neutral','happy','angry','anxiety','sad','surprise', 'hurt'])
                ax1 = ax.twinx()
                ax1.plot(result.posneg, color = 'salmon')
                ax1.set_yticks(['positive','negative'])
                ax1.set_yticklabels(['Positive', 'Negative'])
                st.pyplot(fig)
            count = 0
            lst_all = []
            lst = []
            for idx, i in enumerate(result.posneg):
                # print(i)
                if i == 'negative':
                    count += 1
                    lst.append(idx)
                else:
                    if count >= 5:
                        lst_all.append(deepcopy(lst))
                    count = 0
                    lst = []
            
            if len(lst_all) > 0:
                for seq in lst_all:
                    start = seq[0]
                    end = seq[-1]
                    start_sec = result.loc[start, 'seconds']
                    end_sec = result.loc[end, 'seconds']
                    st.warning(f'{round(start_sec, 2)}초 ~ {round(end_sec, 2)}초의 표정이 좋지 않습니다.')
            else:
                st.success('표정을 잘 지었습니다.')

        # with tab2:
        #     st.header("Pose")
        #     st.subheader("니 자세가 이렇다 삐딱하이 에픽하이")

        #     # pose estimation
        #     pose_video = open(
        #         f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/pose_recording.webm",
        #         "rb",
        #     )
        #     pose_video_bytes = pose_video.read()
        #     st.video(pose_video_bytes)

        #     shoulder_result = pd.read_json(r_shoulder.json(), orient="records")
        #     st.write("SHOULDER")
        #     st.dataframe(shoulder_result)

        #     hand_result = pd.read_json(r_hand.json(), orient="records")
        #     st.write("HAND")
        #     st.dataframe(hand_result)

        # with tab3:
        #     st.header("Eye")
        #     st.subheader("동태눈깔 꼬라보노 보노보노")
        #     st.write("None : 정면 | Side: 그 외")
        #     st.dataframe(eye_result)
        #     video_file = open(
        #         f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/eye_recording.webm",
        #         "rb",
        #     )
        #     video_bytes = video_file.read()
        #     st.video(video_bytes)

    else:
        st.subheader("면접 영상이 제대로 저장되지 않았습니다. 다시 면접 영상을 녹화해주세요.")
else:
    st.subheader("면접 영상이 선택되지 않았습니다. 다시 면접 영상을 녹화해주세요.")
