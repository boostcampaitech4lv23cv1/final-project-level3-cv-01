import os
import sys
import cv2
import ast
from copy import deepcopy

sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

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
assert os.path.exists("./hey-i-375802-d3dcfd2b25d1.json"), "Key가 존재하지 않습니다."

# threshold 지정
emotion_threshold_sec = 1
pose_horizontal_threshold = 0.1
pose_vertical_threshold = 11.4
pose_threshold_sec = 1
eye_threshold_sec = 1

if 'result_dir' in st.session_state.keys():
    if os.path.exists(st.session_state.confirm_video):
        st.subheader("면접 영상 분석 결과입니다.")

        VIDEO_PATH = st.session_state.confirm_video
        result = pd.read_csv("/".join([st.session_state.result_dir, 'result.csv']), index_col=0)
        pose_result = pd.read_csv("/".join([st.session_state.result_dir, 'pose_result.csv']), index_col=0)
        eye_result = pd.read_csv("/".join([st.session_state.result_dir, 'eye_result.csv']), index_col=0)
        tab1, tab2, tab3 = st.tabs(["Emotion", "Pose", "Eye"])

        with tab1:
            st.header("Emotion")
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
            with st.expander('More Information - Facial Emotion Result'):
                st.dataframe(result)

            col1, col2 = st.columns(2)
            with col1:
                linechart = st.selectbox(
                    'What kind of line chart do you want?',
                    ('Emotion (7 classes)', 'Positive or Negative', 'Both')
                    )

                fig, ax = plt.subplots()
                ax.set_xlabel('Time(sec)')
                ax.set_ylabel('Emotion')

                x = np.linspace(0, len(result), 200)

                ax.set_xticks([i for idx, i in enumerate(x) if idx % 15 == 1])
                ax.set_xticklabels([round(i/30, 1) for idx, i in enumerate(x) if idx % 15 == 1])
                ax.set_ylim(-0.5, 6.5)
                ax.tick_params(axis='x', rotation=30)

                numemo = result.emotion.replace(
                    ['angry', 'anxiety', 'sad', 'surprise', 'hurt', 'neutral', 'happy'],
                    [0, 1, 2, 3, 4, 5, 6]
                )
                numposneg = result.posneg.replace(
                    ['positive', 'negative'], [1, 0]
                )

                model_emo = make_interp_spline([i for i in range(len(result))], numemo)
                model_posneg = make_interp_spline([i for i in range(len(result))], numposneg)

                interpol_emo = model_emo(x)
                interpol_posneg = model_posneg(x)

                if linechart == 'Emotion (7 classes)':
                    ax.plot(x, interpol_emo, color = 'skyblue', label = 'emotion')
                    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
                    ax.set_yticklabels(['angry', 'anxiety', 'sad', 'surprise', 'hurt', 'neutral', 'happy'])
                    st.pyplot(fig)

                elif linechart == 'Positive or Negative':
                    ax.plot(x, interpol_posneg, color = 'salmon')
                    ax.set_yticks([1, 0])
                    ax.set_yticklabels(['Positive', 'Negative'])
                    st.pyplot(fig)
                
                elif linechart == 'Both':
                    ax.plot(x, interpol_emo, color = 'skyblue', label = 'emotion')
                    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
                    ax.set_yticklabels(['angry', 'anxiety', 'sad', 'surprise', 'hurt', 'neutral', 'happy'])
                    ax1 = ax.twinx()
                    ax1.plot(x, interpol_posneg, color = 'salmon')
                    ax1.set_yticks([1, 0])
                    ax1.set_yticklabels(['Positive', 'Negative'])
                    st.pyplot(fig)

            with col2:
                count = 0
                lst_all = []
                lst = []
                threshold_sec = emotion_threshold_sec
                threshold = 30 * threshold_sec
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
                        st.warning(f'{round(start_sec, 2)}초 ~ {round(end_sec, 2)}초의 표정이 부정적입니다.')
                else:
                    st.success('표정이 긍정적입니다.')

        with tab2:
            st.header("Pose")

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
                # lst.append((by['right_eye'] - by['left_eye']) / (bx['right_eye'] - bx['left_eye']))
                # lst.append((by['right_ear'] - by['left_ear']) / (bx['right_ear'] - bx['left_ear']))
                # lst.append((by['right_shoulder'] - by['left_shoulder']) / (bx['right_shoulder'] - bx['left_shoulder']))
                # lst.append((by['nose'] - (by['right_shoulder'] + by['left_shoulder']) / 2) / max((bx['nose'] - (bx['right_shoulder'] + bx['left_shoulder']) / 2), 1e-6))
                lst.append((by['right_eye'] - by['left_eye']) / (bx['right_eye'] - bx['left_eye']) if bx['right_eye'] != bx['left_eye'] else 999.)
                lst.append((by['right_ear'] - by['left_ear']) / (bx['right_ear'] - bx['left_ear']) if bx['right_ear'] != bx['left_ear'] else 999.)
                lst.append((by['right_shoulder'] - by['left_shoulder']) / (bx['right_shoulder'] - bx['left_shoulder']) if bx['right_shoulder'] != bx['left_shoulder'] else 999.)
                lst.append((by['nose'] - (by['right_shoulder'] + by['left_shoulder']) / 2) / (bx['nose'] - (bx['right_shoulder'] + bx['left_shoulder']) / 2) if bx['nose'] != (bx['right_shoulder'] + bx['left_shoulder']) / 2 else 999.)
                
                lst.append(((by['right_eye'] + by['left_eye']) / 2 - (by['right_shoulder'] + by['left_shoulder']) / 2) / max(((bx['right_eye'] + bx['left_eye']) / 2 - (bx['right_shoulder'] + bx['left_shoulder']) / 2), 1e-6))
                lst.append(bx['right_wrist'] != -1 and bx['right_elbow'] != -1)
                lst.append(bx['left_wrist'] != -1 and bx['left_elbow'] != -1)
                info.loc[i, :] = lst
            info['seconds'] = pose_sec

            vertical_threshold = pose_vertical_threshold
            horizontal_threshold = pose_horizontal_threshold
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
                
            with st.expander('More Information - Pose Estimation Result'):
                st.subheader('Pose result')
                st.dataframe(pose_result)
                st.subheader('Pose Angle')
                st.dataframe(info)
                st.subheader('Pose is Align?')
                st.dataframe(info_)
                    

            count1, count2, count3, count4 = 0, 0, 0, 0
            lst_all1, lst_all2, lst_all3, lst_all4 = [], [], [], []
            lst1, lst2, lst3, lst4 = [], [], [], []
            threshold_sec = pose_threshold_sec
            threshold = 30 * threshold_sec
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
                if hand:
                    count4 += 1
                    lst4.append(i)
                else:
                    if count4 >= threshold:
                        lst_all4.append(deepcopy(lst4))
                    count4 = 0
                    lst4 = []
            
            tab1_, tab2_, tab3_, tab4_ = st.tabs(["Face Align", "Body Align", "Vertical Align", "Hand"])
            with tab1_:
                if len(lst_all1) > 0:
                    for seq in lst_all1:
                        start = seq[0]
                        end = seq[-1]
                        start_sec = info_.loc[start, 'seconds']
                        end_sec = info_.loc[end, 'seconds']
                        st.warning(f'{round(start_sec, 2)}초 ~ {round(end_sec, 2)}초의 고개가 기울어졌습니다.')
                else:
                    st.success('얼굴이 잘 정렬되어 있습니다.')
            with tab2_:
                if len(lst_all2) > 0:
                    for seq in lst_all2:
                        start = seq[0]
                        end = seq[-1]
                        start_sec = info_.loc[start, 'seconds']
                        end_sec = info_.loc[end, 'seconds']
                        st.warning(f'{round(start_sec, 2)}초 ~ {round(end_sec, 2)}초의 어깨선이 기울어졌습니다.')
                else:
                    st.success('어깨선이 잘 정렬되어 있습니다.')
            with tab3_:
                if len(lst_all3) > 0:
                    for seq in lst_all3:
                        start = seq[0]
                        end = seq[-1]
                        start_sec = info_.loc[start, 'seconds']
                        end_sec = info_.loc[end, 'seconds']
                        st.warning(f'{round(start_sec, 2)}초 ~ {round(end_sec, 2)}초의 몸이 기울어졌습니다.')
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

            with st.expander('More Information - Eye Tracking Result'):
                st.dataframe(eye_result)

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots()
                ax.set_xlabel('Time(sec)')
                ax.set_ylabel('Emotion')

                x = np.linspace(0, len(eye_result), 200)

                ax.set_xticks([i for idx, i in enumerate(x) if idx % 15 == 1])
                ax.set_xticklabels([round(i/30, 1) for idx, i in enumerate(x) if idx % 15 == 1])
                ax.tick_params(axis='x', rotation=30)
                
                numeye = eye_result.tracking.replace(
                    ['Right', 'Center', 'Left'], [-1, 0, 1]
                )

                model_eye = make_interp_spline([i for i in range(len(eye_result))], numeye)

                interpol_eye = model_eye(x)

                ax.plot(x, interpol_eye, color = 'skyblue', label = 'emotion')
                ax.set_ylim(-1.3, 1.3)
                ax.set_yticks([-1, 0, 1])
                ax.set_yticklabels(['Right', 'Center', 'Left'])
                st.pyplot(fig)

            with col2:
                count = 0
                right_lst_all = []
                left_lst_all = []
                right_lst = []
                left_lst = []
                threshold_sec = eye_threshold_sec
                threshold = 30 * threshold_sec
                for idx, i in enumerate(eye_result.tracking):
                    if i == 'Right':
                        count += 1
                        right_lst.append(idx)
                    else:
                        if count >= threshold:
                            right_lst_all.append(deepcopy(right_lst))
                        count = 0
                        right_lst = []
                
                for idx, i in enumerate(eye_result.tracking):
                    if i == 'Left':
                        count += 1
                        left_lst.append(idx)
                    else:
                        if count >= threshold:
                            left_lst_all.append(deepcopy(left_lst))
                        count = 0
                        left_lst = []

                lst_all_dict = {}
                for i in right_lst_all:
                    start = i[0]
                    end = i[-1]
                    lst_all_dict[start] = [end, '오른쪽']
                for i in left_lst_all:
                    start = i[0]
                    end = i[-1]
                    lst_all_dict[start] = [end, '왼쪽']
                lst_all_dict = sorted(lst_all_dict.items())

                if len(lst_all_dict) > 0:
                    for seq, direction in lst_all_dict:
                        start = seq
                        end = direction[0]
                        start_sec = eye_result.loc[start, 'seconds']
                        end_sec = eye_result.loc[end, 'seconds']
                        st.warning(f'{round(start_sec, 2)}초 ~ {round(end_sec, 2)}초의 시선이 {direction[1]}을 응시하고 있습니다.')
                else:
                    st.success('정면을 잘 응시하고 있습니다.')
                

    else:
        st.subheader("면접 영상이 제대로 저장되지 않았습니다. 다시 면접 영상을 녹화해주세요.")
else:
    st.subheader("면접 영상이 선택되지 않았습니다. 다시 면접 영상을 녹화해주세요.")
