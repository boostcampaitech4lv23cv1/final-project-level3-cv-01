import os
import sys
import cv2
import math
from copy import deepcopy

import cv2

sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import make_interp_spline


# 시간 측정

cls_to_idx = {
    "angry":0,
    "anxiety":1,
    "happy":2,
    "blank":3,
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
pose_horizontal_threshold = 5 * (math.pi/180)
pose_vertical_threshold = 85 * (math.pi/180)
pose_threshold_sec = 1
eye_threshold_sec = 1

if 'result_dir' in st.session_state.keys():
    if os.path.exists(st.session_state.confirm_video):
        st.subheader("면접 영상 분석 결과입니다.")
        for name in ['posedb', 'eyedb', 'facedb']:
            if name not in st.session_state:    
                print("DB 요청 실패")
                st.write("DB 요청 실패")
                
        posedb = st.session_state["posedb"]
        eyedb = st.session_state["eyedb"]
        facedb = st.session_state["facedb"]

        VIDEO_PATH = st.session_state.confirm_video
        result = facedb.load_data_inf()
        pose_result = posedb.load_data_inf()
        eye_result = eyedb.load_data_inf()
        
        tab1, tab2, tab3 = st.tabs(["😀 Emotion", "🧘‍♀️ Pose", "👀 Eye"])

        with tab1:
            st.header("Emotion")
            video = cv2.VideoCapture(f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/face_recording.webm")
            video_len = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
            w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
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
                ax.tick_params(axis='x', rotation=30)

                numemo = result.emotion.replace(
                    ['angry', 'anxiety', 'sad', 'surprise', 'blank', 'neutral', 'happy'],
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
                    ax.set_yticklabels(['angry', 'anxiety', 'sad', 'surprise', 'blank', 'neutral', 'happy'])
                    ax.set_ylim(-0.5, 6.5)
                    st.pyplot(fig)

                elif linechart == 'Positive or Negative':
                    ax.plot(x, interpol_posneg, color = 'salmon')
                    ax.set_yticks([1, 0])
                    ax.set_yticklabels(['Positive', 'Negative'])
                    ax.set_ylim(-0.1, 1.1)
                    st.pyplot(fig)
                
                elif linechart == 'Both':
                    ax.plot(x, interpol_emo, color = 'skyblue', label = 'Emotion (7 classes)')
                    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
                    ax.set_yticklabels(['angry', 'anxiety', 'sad', 'surprise', 'blank', 'neutral', 'happy'])
                    ax.set_ylim(-0.5, 6.5)
                    ax1 = ax.twinx()
                    ax1.plot(x, interpol_posneg, color = 'salmon', label='Positive or Negative')
                    ax1.set_yticks([1, 0])
                    ax1.set_yticklabels(['Positive', 'Negative'])
                    ax1.set_ylim(-0.1, 1.1)
                    fig.legend(loc='upper right')
                    st.pyplot(fig)

            with col2:
                lst_all = []
                lst = []
                threshold_sec = emotion_threshold_sec
                threshold = 30 * threshold_sec
                for idx, i in enumerate(result.posneg):
                    if i == 'negative':
                        lst.append(idx)
                    else:
                        if len(lst) >= threshold:
                            lst_all.append(deepcopy(lst))
                        lst = []
                if len(lst) >= threshold:
                    lst_all.append(deepcopy(lst))
                
                st.session_state.face_time = set()
                if len(lst_all) > 0:
                    for seq in lst_all:
                        start = seq[0]
                        end = seq[-1]
                        start_sec = result.loc[start, 'seconds']
                        end_sec = result.loc[end, 'seconds']
                        st.session_state.face_time.add(tuple([start_sec, end_sec, start, end, '_']))
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

            a = pose_result[['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder', 'mid_shoulder','left_elbow','right_elbow','left_wrist','right_wrist']]
            ax = pd.DataFrame(columns = a.columns)
            ay = pd.DataFrame(columns = a.columns)

            for i in range(len(a)):
                info = a.loc[i, :]
                xlst = []
                ylst = []
                for j in info:
                    x, y = j[0], j[1]
                    if x < 0 or x > w:
                        xlst.append(-1)
                        ylst.append(-1)
                    elif y < 0 or y > h:
                        xlst.append(-1)
                        ylst.append(-1)        
                    else:
                        xlst.append(x)
                        ylst.append(y)
                ax.loc[i, :] = xlst
                ay.loc[i, :] = ylst
            st.dataframe(ax)
            st.dataframe(ay)
            info = pd.DataFrame(columns = ['eye-eye','ear-ear','shoulder-shoulder','nose-mid_shoulder', 'eye-mid_shoulder','right_hand-yes','left_hand-yes', 'hand'])
            for i in range(len(a)):
                bx = ax.loc[i,:]
                by = ay.loc[i,:]
                lst = []
                lst.append((by['right_eye'] - by['left_eye']) / (bx['right_eye'] - bx['left_eye']))
                lst.append((by['right_ear'] - by['left_ear']) / (bx['right_ear'] - bx['left_ear']))
                lst.append((by['right_shoulder'] - by['left_shoulder']) / (bx['right_shoulder'] - bx['left_shoulder']))
                lst.append((by['nose'] - by['mid_shoulder']) / max((bx['nose'] - bx['mid_shoulder']), 1e-6))
                lst.append(((by['right_eye'] + by['left_eye']) / 2 - by['mid_shoulder']) / max(((bx['right_eye'] + bx['left_eye']) / 2 - bx['mid_shoulder']), 1e-6))
                right_hand = (bx['right_wrist'] != -1) and (bx['right_elbow'] != -1)
                left_hand = (bx['left_wrist'] != -1) and (bx['left_elbow'] != -1)
                lst.append(right_hand)
                lst.append(left_hand)
                if right_hand and left_hand:
                    lst.append('both')
                elif right_hand and not left_hand:
                    lst.append('right')
                elif left_hand and not right_hand:
                    lst.append('left')
                else:
                    lst.append('none')
                info.loc[i, :] = lst
            
            info['seconds'] = pose_sec

            vertical_threshold = np.tan(pose_vertical_threshold)
            horizontal_threshold = np.tan(pose_horizontal_threshold)
            info_ = pd.DataFrame(columns = ['face_align', 'body_align', 'vertical_align', 'hand', 'seconds'])
            for i in range(len(info)):
                lst = []
                eye_eye, ear_ear, shd_shd, nose_chest, eye_chest, rhand, lhand, hand, secs = info.loc[i, :]
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
            
            col1, col2 = st.columns(2)
            with col1:
                mode = st.selectbox(
                    'What kind of line chart do you want?',
                    ('Eye-Eye', 'Ear-Ear', 'Shoulder-Shoulder', 'Nose-Mid Shoulder', 'Eye-Mid Shoulder', 'Horizontal', 'Vertical', 'Hand')
                    )
                
                x = np.linspace(0, len(result), 200)

                if mode == 'Eye-Eye':
                    fig, ax = plt.subplots()
                    ax.set_xlabel('Time(sec)')
                    ax.set_ylabel('Angle')

                    ax.set_xticks([i for idx, i in enumerate(x) if idx % 15 == 1])
                    ax.set_xticklabels([round(i/30, 1) for idx, i in enumerate(x) if idx % 15 == 1])
                    ax.tick_params(axis='x', rotation=30)
                    angle_y = [i * (180 / math.pi) for i in np.arctan(info['eye-eye'].astype(np.float64))]
                    ax.axhline(y = pose_horizontal_threshold * (180 / math.pi), color='lightcoral', linestyle='--', alpha=0.5)
                    ax.axhline(y = - pose_horizontal_threshold * (180 / math.pi), color='lightcoral', linestyle='--', alpha=0.5)
                    ax.axhline(y = 0, color='lightcoral', linestyle='--', alpha=0.5)
                    ax.plot(angle_y, color='skyblue')
                    st.pyplot(fig)
                
                elif mode == 'Ear-Ear':
                    fig, ax = plt.subplots()
                    ax.set_xlabel('Time(sec)')
                    ax.set_ylabel('Angle')

                    ax.set_xticks([i for idx, i in enumerate(x) if idx % 15 == 1])
                    ax.set_xticklabels([round(i/30, 1) for idx, i in enumerate(x) if idx % 15 == 1])
                    ax.tick_params(axis='x', rotation=30)
                    
                    angle_y = [i * (180 / math.pi) for i in np.arctan(info['ear-ear'].astype(np.float64))]
                    ax.axhline(y = pose_horizontal_threshold * (180 / math.pi), color='lightcoral', linestyle='--', alpha=0.5)
                    ax.axhline(y = - pose_horizontal_threshold * (180 / math.pi), color='lightcoral', linestyle='--', alpha=0.5)
                    ax.axhline(y = 0, color='lightcoral', linestyle='--', alpha=0.5)
                    ax.plot(angle_y, color='skyblue')
                    st.pyplot(fig)

                elif mode == 'Shoulder-Shoulder':
                    fig, ax = plt.subplots()
                    ax.set_xlabel('Time(sec)')
                    ax.set_ylabel('Angle')

                    ax.set_xticks([i for idx, i in enumerate(x) if idx % 15 == 1])
                    ax.set_xticklabels([round(i/30, 1) for idx, i in enumerate(x) if idx % 15 == 1])
                    ax.tick_params(axis='x', rotation=30)
                    
                    angle_y = [i * (180 / math.pi) for i in np.arctan(info['shoulder-shoulder'].astype(np.float64))]
                    ax.axhline(y = pose_horizontal_threshold * (180 / math.pi), color='lightcoral', linestyle='--', alpha=0.5)
                    ax.axhline(y = - pose_horizontal_threshold * (180 / math.pi), color='lightcoral', linestyle='--', alpha=0.5)
                    ax.axhline(y = 0, color='lightcoral', linestyle='--', alpha=0.5)
                    ax.plot(angle_y, color='skyblue')
                    st.pyplot(fig)
                
                elif mode == 'Nose-Mid Shoulder':
                    fig, ax = plt.subplots()
                    ax.set_xlabel('Time(sec)')
                    ax.set_ylabel('Angle')

                    ax.set_xticks([i for idx, i in enumerate(x) if idx % 15 == 1])
                    ax.set_xticklabels([round(i/30, 1) for idx, i in enumerate(x) if idx % 15 == 1])
                    ax.tick_params(axis='x', rotation=30)
                    
                    angle_y = [i * (180 / math.pi) for i in np.arctan(info['nose-mid_shoulder'].astype(np.float64))]
                    ax.axhline(y = pose_vertical_threshold * (180 / math.pi), color='lightcoral', linestyle='--', alpha=0.5)
                    ax.axhline(y = 180 - pose_vertical_threshold * (180 / math.pi), color='lightcoral', linestyle='--', alpha=0.5)
                    ax.axhline(y = 90, color='lightcoral', linestyle='--', alpha=0.5)
                    ax.plot([i if i > 0 else i + 180 for i in angle_y], color='skyblue')
                    ax.set_ylim(70, 110)
                    st.pyplot(fig)

                elif mode == 'Eye-Mid Shoulder':
                    fig, ax = plt.subplots()
                    ax.set_xlabel('Time(sec)')
                    ax.set_ylabel('Angle')

                    ax.set_xticks([i for idx, i in enumerate(x) if idx % 15 == 1])
                    ax.set_xticklabels([round(i/30, 1) for idx, i in enumerate(x) if idx % 15 == 1])
                    ax.tick_params(axis='x', rotation=30)
                    
                    angle_y = [i * (180 / math.pi) for i in np.arctan(info['eye-mid_shoulder'].astype(np.float64))]
                    ax.axhline(y = pose_vertical_threshold * (180 / math.pi), color='lightcoral', linestyle='--', alpha=0.5)
                    ax.axhline(y = 180 - pose_vertical_threshold * (180 / math.pi), color='lightcoral', linestyle='--', alpha=0.5)
                    ax.axhline(y = 90, color='lightcoral', linestyle='--', alpha=0.5)
                    ax.plot([i if i > 0 else i + 180 for i in angle_y], color='skyblue')
                    ax.set_ylim(70, 110)
                    st.pyplot(fig)

                elif mode == 'Horizontal':
                    fig, ax = plt.subplots()
                    ax.set_xlabel('Time(sec)')
                    ax.set_ylabel('Angle')

                    ax.set_xticks([i for idx, i in enumerate(x) if idx % 15 == 1])
                    ax.set_xticklabels([round(i/30, 1) for idx, i in enumerate(x) if idx % 15 == 1])
                    ax.tick_params(axis='x', rotation=30)
                    
                    angle_y1 = [i * (180 / math.pi) for i in np.arctan(info['eye-eye'].astype(np.float64))]
                    angle_y2 = [i * (180 / math.pi) for i in np.arctan(info['ear-ear'].astype(np.float64))]
                    angle_y3 = [i * (180 / math.pi) for i in np.arctan(info['shoulder-shoulder'].astype(np.float64))]
                    ax.axhline(y = pose_horizontal_threshold * (180 / math.pi), color='lightcoral', linestyle='--', alpha=0.5)
                    ax.axhline(y = - pose_horizontal_threshold * (180 / math.pi), color='lightcoral', linestyle='--', alpha=0.5)
                    ax.axhline(y = 0, color='lightcoral', linestyle='--', alpha=0.5)
                    ax.plot(angle_y1, color='skyblue', label='Eye-Eye')
                    ax.plot(angle_y2, color='yellowgreen', label='Ear-Ear')
                    ax.plot(angle_y3, color='khaki', label='Shoulder-Shoulder')
                    ax.legend(loc='best')
                    st.pyplot(fig)

                elif mode == 'Vertical':
                    fig, ax = plt.subplots()
                    ax.set_xlabel('Time(sec)')
                    ax.set_ylabel('Angle')

                    ax.set_xticks([i for idx, i in enumerate(x) if idx % 15 == 1])
                    ax.set_xticklabels([round(i/30, 1) for idx, i in enumerate(x) if idx % 15 == 1])
                    ax.tick_params(axis='x', rotation=30)

                    angle_y1 = [i * (180 / math.pi) for i in np.arctan(info['nose-mid_shoulder'].astype(np.float64))]
                    angle_y2 = [i * (180 / math.pi) for i in np.arctan(info['eye-mid_shoulder'].astype(np.float64))]
                    ax.axhline(y = pose_vertical_threshold * (180 / math.pi), color='lightcoral', linestyle='--', alpha=0.5)
                    ax.axhline(y = 180 - pose_vertical_threshold * (180 / math.pi), color='lightcoral', linestyle='--', alpha=0.5)
                    ax.axhline(y = 90, color='lightcoral', linestyle='--', alpha=0.5)
                    ax.plot([i if i > 0 else i + 180 for i in angle_y1], color='skyblue', label='Nose-Mid Shoulder')
                    ax.plot([i if i > 0 else i + 180 for i in angle_y2], color='yellowgreen', label='Eye-Mid Shoulder')
                    ax.set_ylim(70, 110)
                    ax.legend(loc='best')
                    st.pyplot(fig)
                    
                elif mode == 'Hand':
                    numhand = info.hand.replace(
                        ['none', 'left', 'right', 'both'], [0, 1, 2, 3]
                    )

                    fig, ax = plt.subplots()
                    ax.set_xlabel('Time(sec)')
                    ax.set_ylabel('Hand or No')

                    ax.set_xticks([i for idx, i in enumerate(x) if idx % 15 == 1])
                    ax.set_xticklabels([round(i/30, 1) for idx, i in enumerate(x) if idx % 15 == 1])
                    ax.set_yticks([0, 1, 2, 3])
                    ax.set_yticklabels(['No Hand', 'Left Hand', 'Right Hand', 'Both Hand'])
                    ax.tick_params(axis='x', rotation=30)
                    ax.plot(numhand, color='skyblue', label='Nose-Mid Shoulder')
                    st.pyplot(fig)

            with col2:
                lst_all1, lst_all2, lst_all3, lst_all4 = [], [], [], []
                lst1, lst2, lst3, lst4 = [], [], [], []
                threshold_sec = pose_threshold_sec
                threshold = 30 * threshold_sec
                for i in range(len(info_)):
                    face, body, vert, hand, _ = info_.loc[i, :]
                    if not face:
                        lst1.append(i)
                    else:
                        if len(lst1) >= threshold:
                            lst_all1.append(deepcopy(lst1))
                        lst1 = []
                    if not body:
                        lst2.append(i)
                    else:
                        if len(lst2) >= threshold:
                            lst_all2.append(deepcopy(lst2))
                        lst2 = []
                    if not vert:
                        lst3.append(i)
                    else:
                        if len(lst3) >= threshold:
                            lst_all3.append(deepcopy(lst3))
                        lst3 = []
                    if hand:
                        lst4.append(i)
                    else:
                        if len(lst4) >= threshold:
                            lst_all4.append(deepcopy(lst4))
                        lst4 = []
                if len(lst1) >= threshold:
                    lst_all1.append(deepcopy(lst1))
                if len(lst2) >= threshold:
                    lst_all2.append(deepcopy(lst2))
                if len(lst3) >= threshold:
                    lst_all3.append(deepcopy(lst3))
                if len(lst4) >= threshold:
                    lst_all4.append(deepcopy(lst4))
                
                tab1_, tab2_, tab3_, tab4_ = st.tabs(["Face Align", "Body Align", "Vertical Align", "Hand"])
                st.session_state.pose_time = set()
                with tab1_:
                    if len(lst_all1) > 0:
                        for seq in lst_all1:
                            start = seq[0]
                            end = seq[-1]
                            start_sec = info_.loc[start, 'seconds']
                            end_sec = info_.loc[end, 'seconds']
                            st.session_state.pose_time.add(tuple([start_sec, end_sec, start, end, 'face']))
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
                            st.session_state.pose_time.add(tuple([start_sec, end_sec, start, end, 'shoulder']))
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
                            st.session_state.pose_time.add(tuple([start_sec, end_sec, start, end, 'body']))
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
                            st.session_state.pose_time.add(tuple([start_sec, end_sec, start, end, 'hand']))
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
                
                numeye = numeye.replace('None', method='ffill')
                model_eye = make_interp_spline([i for i in range(len(eye_result))], numeye)

                interpol_eye = model_eye(x)

                ax.plot(x, interpol_eye, color = 'skyblue', label = 'emotion')
                ax.set_ylim(-1.3, 1.3)
                ax.set_yticks([-1, 0, 1])
                ax.set_yticklabels(['Right', 'Center', 'Left'])
                ax.axhline(y = 0, color='lightcoral', linestyle='--', alpha=0.5)
                st.pyplot(fig)

            with col2:
                right_lst_all = []
                left_lst_all = []
                right_lst = []
                left_lst = []
                threshold_sec = eye_threshold_sec
                threshold = 30 * threshold_sec
                for idx, i in enumerate(eye_result.tracking):
                    if i == 'Right':
                        right_lst.append(idx)
                    else:
                        if len(right_lst) >= threshold:
                            right_lst_all.append(deepcopy(right_lst))
                        right_lst = []

                if len(right_lst) >= threshold:
                    right_lst_all.append(deepcopy(right_lst))

                for idx, i in enumerate(eye_result.tracking):
                    if i == 'Left':
                        left_lst.append(idx)
                    else:
                        if len(left_lst) >= threshold:
                            left_lst_all.append(deepcopy(left_lst))
                        left_lst = []
                
                if len(left_lst) >= threshold:
                    left_lst_all.append(deepcopy(left_lst))

                lst_all_dict = {}

                if len(right_lst_all) > 0:
                    for i in right_lst_all:
                        start = i[0]
                        end = i[-1]
                        lst_all_dict[start] = [end, '오른쪽']

                if len(left_lst_all) > 0:
                    for i in left_lst_all:
                        start = i[0]
                        end = i[-1]
                        lst_all_dict[start] = [end, '왼쪽']
                lst_all_dict = sorted(lst_all_dict.items())

                st.session_state.eye_time = set()

                if len(lst_all_dict) > 0:
                    for seq, direction in lst_all_dict:
                        start = seq
                        end = direction[0]
                        start_sec = eye_result.loc[start, 'seconds']
                        end_sec = eye_result.loc[end, 'seconds']
                        st.session_state.eye_time.add(tuple([start_sec, end_sec, start, end, direction[1]]))
                        st.warning(f'{round(start_sec, 2)}초 ~ {round(end_sec, 2)}초의 시선이 {direction[1]}을 응시하고 있습니다.')
                else:
                    st.success('정면을 잘 응시하고 있습니다.')
                

    else:
        st.subheader("면접 영상이 제대로 저장되지 않았습니다. 다시 면접 영상을 녹화해주세요.")
else:
    st.subheader("면접 영상이 선택되지 않았습니다. 다시 면접 영상을 녹화해주세요.")
