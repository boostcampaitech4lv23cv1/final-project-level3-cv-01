import os
import sys
import cv2
import ast
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

st.set_page_config(layout="wide")
st.title("HEY-I")

# key 존재 확인
assert os.path.exists("./hey-i-375802-e6e402d22694.json"), "Key가 존재하지 않습니다."


# st.session_state.result_dir = "./최명헌_5126/230204_021257"
# st.session_state.confirm_video = st.session_state.result_dir

if 'result_dir' in st.session_state.keys():
    if os.path.exists(st.session_state.confirm_video):
        st.subheader("면접 영상 분석 결과입니다.")

        VIDEO_PATH = st.session_state.confirm_video
        result = pd.read_csv(os.path.join(st.session_state.result_dir, 'result.csv'), index_col=0)
        pose_result = pd.read_csv(os.path.join(st.session_state.result_dir, 'pose_result.csv'), index_col=0)
        eye_result = pd.read_csv(os.path.join(st.session_state.result_dir, 'eye_result.csv'), index_col=0)

        # VIDEO_PATH = st.session_state.confirm_video
        # result = pd.read_csv(os.path.join(st.session_state.result_dir, 'result.csv'), index_col=0)
        # pose_result = pd.read_csv(os.path.join(st.session_state.result_dir, 'pose_result.csv'), index_col=0)
        # eye_result = pd.read_csv(os.path.join(st.session_state.result_dir, 'eye_result.csv'), index_col=0)
        tab1, tab2, tab3 = st.tabs(["Emotion", "Pose", "Eye"])

        with tab1:
            st.header("Emotion")
            # st.subheader("니 얼굴 표정 이렇다 임마 표정 좀 풀어라")
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
            ax.set_xticks([i+1 for i in range(len(result)) if i%20 == 0])
            ax.set_xticklabels([round(j, 1) for i, j in enumerate(result.seconds) if i%20==0])
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
            threshold_sec = 0.4
            threshold = 20 * threshold_sec
            for idx, i in enumerate(result.posneg):
                # print(i)
                if i == 'negative':
                    count += 1
                    lst.append(idx)
                else:
                    if count >= threshold:
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

        with tab2:
            st.header("Pose")
            # st.subheader("니 자세가 이렇다 삐딱하이 에픽하이")

            pose_video = cv2.VideoCapture(f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/pose_recording.webm")
            pose_video_len = pose_video.get(cv2.CAP_PROP_FRAME_COUNT) / pose_video.get(cv2.CAP_PROP_FPS)
            pose_sec = [pose_video_len / len(pose_result) * (i + 1) for i in range(len(pose_result))]
            pose_result['seconds'] = pose_sec
            pose_video_file = open(
                f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/pose_recording.webm",
                "rb",
            )
            pose_video_bytes = pose_video_file.read()
            st.video(pose_video_bytes)

            st.dataframe(pose_result)

            a = pose_result[['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist']]
            ax = pd.DataFrame(columns = a.columns)
            ay = pd.DataFrame(columns = a.columns)

            for i in range(len(a)):
                info = a.loc[i, :]
                xlst = []
                ylst = []
                for j in info:
                    x, y = ast.literal_eval(j)
                    if x < 0 or x > 640:
                        xlst.append(-1)
                        ylst.append(-1)
                    elif y < 0 or y > 640:
                        xlst.append(-1)
                        ylst.append(-1)        
                    else:
                        xlst.append(x)
                        ylst.append(y)
                ax.loc[i, :] = xlst
                ay.loc[i, :] = ylst
            info = pd.DataFrame(columns = ['eye-eye','ear-ear','shoulder-shoulder','nose-chest', 'eye-chest','right_hand-yes','left_hand-yes'])
            for i in range(len(a)):
                bx = ax.loc[i,:]
                by = ay.loc[i,:]
                lst = []
                lst.append((by['right_eye'] - by['left_eye']) / (bx['right_eye'] - bx['left_eye']))
                lst.append((by['right_ear'] - by['left_ear']) / (bx['right_ear'] - bx['left_ear']))
                lst.append((by['right_shoulder'] - by['left_shoulder']) / (bx['right_shoulder'] - bx['left_shoulder']))
                lst.append((by['nose'] - (by['right_shoulder'] + by['left_shoulder']) / 2) / max((bx['nose'] - (bx['right_shoulder'] + bx['left_shoulder']) / 2), 1e-6))
                lst.append(((by['right_eye'] + by['left_eye']) / 2 - (by['right_shoulder'] + by['left_shoulder']) / 2) / max(((bx['right_eye'] + bx['left_eye']) / 2 - (bx['right_shoulder'] + bx['left_shoulder']) / 2), 1e-6))
                lst.append(bx['right_wrist'] != -1)
                lst.append(bx['left_wrist'] != -1)
                info.loc[i, :] = lst
            info['seconds'] = pose_sec
            st.dataframe(info)

            horizontal_threshold = 0.1
            vertical_threshold = 11.4
            info_ = pd.DataFrame(columns = ['face_align', 'body_align', 'vertical_align', 'hand', 'seconds'])
            for i in range(len(info)):
                lst = []
                eye_eye, ear_ear, shd_shd, nose_chest, eye_chest, rhand, lhand, secs = info.loc[i, :]
                # 얼굴 align
                if abs(eye_eye) < horizontal_threshold or abs(ear_ear) < horizontal_threshold: lst.append(True)
                else: lst.append(False)
                # 몸통 align
                if abs(shd_shd) < horizontal_threshold: lst.append(True)
                else: lst.append(False)
                # 얼굴-몸통 삐딱
                if abs(nose_chest) > vertical_threshold or abs(eye_chest) > vertical_threshold: lst.append(True)
                else: lst.append(False)
                # 손 출현
                if rhand or lhand: lst.append(True)
                else: lst.append(False)
                lst.append(secs)
                info_.loc[i, :] = lst
            st.dataframe(info_)

            count1, count2, count3, count4 = 0, 0, 0, 0
            lst_all1, lst_all2, lst_all3, lst_all4 = [], [], [], []
            lst1, lst2, lst3, lst4 = [], [], [], []
            threshold_sec = 1
            threshold = 20 * threshold_sec
            for i in range(len(info_)):
                face, body, vert, hand, _ = info_.loc[i, :]
                if not face:
                    count1 += 1
                    lst1.append(i)
                else:
                    if count1 >= threshold:
                        lst_all1.append(deepcopy(lst1))
                    count1 = 0
                    lst1 = []
                if not body:
                    count2 += 1
                    lst2.append(i)
                else:
                    if count2 >= threshold:
                        lst_all2.append(deepcopy(lst2))
                    count2 = 0
                    lst2 = []
                if not vert:
                    count3 += 1
                    lst3.append(i)
                else:
                    if count3 >= threshold:
                        lst_all3.append(deepcopy(lst3))
                    count3 = 0
                    lst3 = []
                if not hand:
                    count4 += 1
                    lst4.append(i)
                else:
                    if count4 >= threshold:
                        lst_all4.append(deepcopy(lst4))
                    count4 = 0
                    lst4 = []
            
            tab1_, tab2_, tab3_, tab4_ = st.tabs(["Face Align", "Body Align", "Vertical_Align", "Hand"])
            with tab1_:
                if len(lst_all1) > 0:
                    for seq in lst_all1:
                        start = seq[0]
                        end = seq[-1]
                        start_sec = info_.loc[start, 'seconds']
                        end_sec = info_.loc[end, 'seconds']
                        st.warning(f'{round(start_sec, 2)}초 ~ {round(end_sec, 2)}초의 얼굴이 비뚫어졌습니다.')
                else:
                    st.success('얼굴이 잘 정렬되어 있습니다.')
            with tab2_:
                if len(lst_all2) > 0:
                    for seq in lst_all2:
                        start = seq[0]
                        end = seq[-1]
                        start_sec = info_.loc[start, 'seconds']
                        end_sec = info_.loc[end, 'seconds']
                        st.warning(f'{round(start_sec, 2)}초 ~ {round(end_sec, 2)}초의 몸(어깨선)이 비뚫어졌습니다.')
                else:
                    st.success('몸(어깨선)이 잘 정렬되어 있습니다.')
            with tab3_:
                if len(lst_all3) > 0:
                    for seq in lst_all3:
                        start = seq[0]
                        end = seq[-1]
                        start_sec = info_.loc[start, 'seconds']
                        end_sec = info_.loc[end, 'seconds']
                        st.warning(f'{round(start_sec, 2)}초 ~ {round(end_sec, 2)}초의 몸과 얼굴이 비뚫어졌습니다.')
                else:
                    st.success('몸과 얼굴이 잘 정렬되어 있습니다.')
            with tab4_:
                if len(lst_all4) > 0:
                    for seq in lst_all4:
                        start = seq[0]
                        end = seq[-1]
                        start_sec = info_.loc[start, 'seconds']
                        end_sec = info_.loc[end, 'seconds']
                        st.warning(f'{round(start_sec, 2)}초 ~ {round(end_sec, 2)}초에 손이 나왔습니다.')
                else:
                    st.success('손이 나오지 않았습니다.')


        with tab3:
            st.header("Eye")
            # st.subheader("동태눈깔 꼬라보노 보노보노")
            eye_video = cv2.VideoCapture(f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/eye_recording.webm")
            eye_video_len = eye_video.get(cv2.CAP_PROP_FRAME_COUNT) / max(eye_video.get(cv2.CAP_PROP_FPS), 1e-6)
            eye_sec = [eye_video_len / len(eye_result) * (i + 1) for i in range(len(eye_result))]
            eye_result['seconds'] = eye_sec
            eye_video_file = open(
                f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/eye_recording.webm",
                "rb",
            )
            eye_video_bytes = eye_video_file.read()
            st.video(eye_video_bytes)

            st.dataframe(eye_result.replace('None', method='bfill'))
            
            # eye_result = eye_result.fillna('bfill')

    else:
        st.subheader("면접 영상이 제대로 저장되지 않았습니다. 다시 면접 영상을 녹화해주세요.")
else:
    st.subheader("면접 영상이 선택되지 않았습니다. 다시 면접 영상을 녹화해주세요.")
