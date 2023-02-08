import os
import sys
import cv2
import math

sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from moviepy.editor import VideoFileClip

from DBconnect.feedback import FeedbackDB

# 시간 측정


def slice_video(root_dir, frame_sec_list, type):
    if not os.path.exists("/".join([root_dir, "slice"])):
        os.makedirs("/".join([root_dir, "slice"]))
    vid = VideoFileClip("/".join([root_dir, f"{type}_recording.webm"]))
    for i, (start_sec, end_sec, start, end, _) in enumerate(frame_sec_list):
        print(start, end)
        st.session_state.is_okay[f"{type}_{i}_{start}_{end}"] = False
        if not os.path.exists("/".join([root_dir, "slice", f"{type}_slice_{i}.webm"])):
            vid.subclip(start_sec, end_sec).write_videofile(
                "/".join([root_dir, "slice", f"{type}_slice_{i}.webm"])
            )
    vid.close()


def st_show_video(video_path):
    video_file = open(video_path, "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)


cls_to_idx = {
    "angry": 0,
    "anxiety": 1,
    "happy": 2,
    "blank": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6,
}

elapsed_time = dict()

if not "name" in st.session_state.keys():
    st.warning("HEY-I 페이지에서 이름과 번호를 입력하세요")
    st.stop()

st.set_page_config(layout="wide")
st.title("HEY-I")

# key 존재 확인
assert os.path.exists("./hey-i-375802-d3dcfd2b25d1.json"), "Key가 존재하지 않습니다."

# threshold 지정
emotion_threshold_sec = 1
pose_horizontal_threshold = 5 * (math.pi / 180)
pose_vertical_threshold = 85 * (math.pi / 180)
pose_threshold_sec = 1
eye_threshold_sec = 1

if "face_time" in st.session_state.keys():
    if os.path.exists(st.session_state.confirm_video):
        st.subheader("면접 영상 분석에 대해 확인하고 피드백해주세요.")

        for name in ["posedb", "eyedb", "facedb"]:
            if name not in st.session_state:
                print("DB 요청 실패")
                st.write("DB 요청 실패")

        posedb = st.session_state["posedb"]
        eyedb = st.session_state["eyedb"]
        facedb = st.session_state["facedb"]

        if not os.path.exists("/".join([st.session_state.result_dir, "slice"])):
            cut_video = st.button("More Information")
            if cut_video:
                with st.spinner("더 많은 정보를 위해 추가적인 데이터를 생성하고 있습니다..."):
                    st.session_state.is_okay = {}
                    slice_video(
                        st.session_state.result_dir,
                        sorted(st.session_state.face_time),
                        "face",
                    )
                    slice_video(
                        st.session_state.result_dir,
                        sorted(st.session_state.pose_time),
                        "pose",
                    )
                    slice_video(
                        st.session_state.result_dir,
                        sorted(st.session_state.eye_time),
                        "eye",
                    )

        else:
            # st.write(st.session_state.is_okay)
            VIDEO_PATH = st.session_state.confirm_video
            video = cv2.VideoCapture(
                f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/face_recording.webm"
            )
            video_len = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(
                cv2.CAP_PROP_FPS
            )
            w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            result = facedb.load_data_inf()
            pose_result = posedb.load_data_inf()
            eye_result = eyedb.load_data_inf()
            tab1, tab2, tab3 = st.tabs(["😀 Emotion", "🧘‍♀️ Pose", "👀 Eye"])

            with tab1:
                st.header("Emotion")
                video = cv2.VideoCapture(
                    f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/face_recording.webm"
                )
                video_len = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(
                    cv2.CAP_PROP_FPS
                )
                sec = [video_len / len(result) * (i + 1) for i in range(len(result))]
                result["seconds"] = sec

                with st.expander("More Information - Facial Emotion Result"):
                    st.dataframe(result)

                x = np.linspace(0, len(result), 200)

                numemo = result.emotion.replace(
                    [
                        "angry",
                        "anxiety",
                        "sad",
                        "surprise",
                        "blank",
                        "neutral",
                        "happy",
                    ],
                    [0, 1, 2, 3, 4, 5, 6],
                )
                numposneg = result.posneg.replace(["positive", "negative"], [1, 0])

                model_emo = make_interp_spline([i for i in range(len(result))], numemo)
                model_posneg = make_interp_spline(
                    [i for i in range(len(result))], numposneg
                )

                interpol_emo = model_emo(x)
                interpol_posneg = model_posneg(x)

                st.subheader("구간 별 동영상을 확인해보세요")
                if len(st.session_state.face_time) > 0:
                    for idx, (start_sec, end_sec, start, end, _) in enumerate(
                        sorted(st.session_state.face_time)
                    ):
                        with st.expander(
                            f"🔴 {round(start_sec, 2)}초 ~ {round(end_sec, 2)}초의 표정이 부정적입니다."
                        ):
                            col1, col2 = st.columns(2)
                            with col1:
                                linechart = st.selectbox(
                                    "What kind of line chart do you want?",
                                    (
                                        "Emotion (7 classes)",
                                        "Positive or Negative",
                                        "Both",
                                    ),
                                    key=idx,
                                )
                                fig, ax = plt.subplots()
                                ax.set_xlabel("Time(sec)")
                                ax.set_ylabel("Emotion")

                                ax.set_xticks(
                                    [i for idx, i in enumerate(x) if idx % 15 == 1]
                                )
                                ax.set_xticklabels(
                                    [
                                        round(i / 30, 1)
                                        for idx, i in enumerate(x)
                                        if idx % 15 == 1
                                    ]
                                )
                                ax.tick_params(axis="x", rotation=30)

                                if linechart == "Emotion (7 classes)":
                                    ax.plot(
                                        x,
                                        interpol_emo,
                                        color="skyblue",
                                        label="emotion",
                                    )
                                    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
                                    ax.set_yticklabels(
                                        [
                                            "angry",
                                            "anxiety",
                                            "sad",
                                            "surprise",
                                            "blank",
                                            "neutral",
                                            "happy",
                                        ]
                                    )
                                    ax.set_ylim(-0.5, 6.5)

                                elif linechart == "Positive or Negative":
                                    ax.plot(x, interpol_posneg, color="salmon")
                                    ax.set_yticks([1, 0])
                                    ax.set_yticklabels(["Positive", "Negative"])
                                    ax.set_ylim(-0.1, 1.1)

                                elif linechart == "Both":
                                    ax.plot(
                                        x,
                                        interpol_emo,
                                        color="skyblue",
                                        label="Emotion (7 classes)",
                                    )
                                    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
                                    ax.set_yticklabels(
                                        [
                                            "angry",
                                            "anxiety",
                                            "sad",
                                            "surprise",
                                            "blank",
                                            "neutral",
                                            "happy",
                                        ]
                                    )
                                    ax.set_ylim(-0.5, 6.5)
                                    ax1 = ax.twinx()
                                    ax1.plot(
                                        x,
                                        interpol_posneg,
                                        color="salmon",
                                        label="Positive or Negative",
                                    )
                                    ax1.set_yticks([1, 0])
                                    ax1.set_yticklabels(["Positive", "Negative"])
                                    ax1.set_ylim(-0.1, 1.1)
                                    fig.legend(loc="upper right")

                                ax.axvline(
                                    x=start_sec * 30,
                                    linestyle="--",
                                    color="black",
                                    alpha=0.5,
                                )
                                ax.axvline(
                                    x=end_sec * 30,
                                    linestyle="--",
                                    color="black",
                                    alpha=0.5,
                                )
                                st.pyplot(fig)

                            with col2:
                                st_show_video(
                                    "/".join(
                                        [
                                            st.session_state.result_dir,
                                            "slice",
                                            f"face_slice_{idx}.webm",
                                        ]
                                    )
                                )
                                st.session_state.is_okay[
                                    f"face_{idx}_{start}_{end}"
                                ] = st.checkbox("👌 이 분석 결과에 만족하지 않으신가요?", key=idx + 100)
                                if st.session_state.is_okay[
                                    f"face_{idx}_{start}_{end}"
                                ]:
                                    st.success("감사합니다.", icon="🔥")
                else:
                    with st.expander("🟢 표정이 긍정적입니다."):
                        col1, col2 = st.columns(2)
                        with col1:
                            linechart = st.selectbox(
                                "What kind of line chart do you want?",
                                ("Emotion (7 classes)", "Positive or Negative", "Both"),
                                key=10000,
                            )
                            fig, ax = plt.subplots()
                            ax.set_xlabel("Time(sec)")
                            ax.set_ylabel("Emotion")

                            ax.set_xticks(
                                [i for idx, i in enumerate(x) if idx % 15 == 1]
                            )
                            ax.set_xticklabels(
                                [
                                    round(i / 30, 1)
                                    for idx, i in enumerate(x)
                                    if idx % 15 == 1
                                ]
                            )
                            ax.tick_params(axis="x", rotation=30)

                            if linechart == "Emotion (7 classes)":
                                ax.plot(
                                    x, interpol_emo, color="skyblue", label="emotion"
                                )
                                ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
                                ax.set_yticklabels(
                                    [
                                        "angry",
                                        "anxiety",
                                        "sad",
                                        "surprise",
                                        "blank",
                                        "neutral",
                                        "happy",
                                    ]
                                )
                                ax.set_ylim(-0.5, 6.5)

                            elif linechart == "Positive or Negative":
                                ax.plot(x, interpol_posneg, color="salmon")
                                ax.set_yticks([1, 0])
                                ax.set_yticklabels(["Positive", "Negative"])
                                ax.set_ylim(-0.1, 1.1)

                            elif linechart == "Both":
                                ax.plot(
                                    x,
                                    interpol_emo,
                                    color="skyblue",
                                    label="Emotion (7 classes)",
                                )
                                ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
                                ax.set_yticklabels(
                                    [
                                        "angry",
                                        "anxiety",
                                        "sad",
                                        "surprise",
                                        "blank",
                                        "neutral",
                                        "happy",
                                    ]
                                )
                                ax.set_ylim(-0.5, 6.5)
                                ax1 = ax.twinx()
                                ax1.plot(
                                    x,
                                    interpol_posneg,
                                    color="salmon",
                                    label="Positive or Negative",
                                )
                                ax1.set_yticks([1, 0])
                                ax1.set_yticklabels(["Positive", "Negative"])
                                ax1.set_ylim(-0.1, 1.1)
                                fig.legend(loc="upper right")

                            st.pyplot(fig)

                        with col2:
                            st.session_state.is_okay["face_all"] = st.checkbox(
                                "👌 이 분석 결과에 만족하지 않으신가요?", key=100000
                            )
                            if st.session_state.is_okay["face_all"]:
                                st.success("감사합니다.", icon="🔥")

            with tab2:
                st.header("Pose")

                pose_video = cv2.VideoCapture(
                    f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/pose_recording.webm"
                )
                pose_video_len = pose_video.get(
                    cv2.CAP_PROP_FRAME_COUNT
                ) / pose_video.get(cv2.CAP_PROP_FPS)
                pose_sec = [
                    pose_video_len / len(pose_result) * (i + 1)
                    for i in range(len(pose_result))
                ]
                pose_result["seconds"] = pose_sec

                a = pose_result[
                    [
                        "nose",
                        "left_eye",
                        "right_eye",
                        "left_ear",
                        "right_ear",
                        "left_shoulder",
                        "right_shoulder",
                        "mid_shoulder",
                        "left_elbow",
                        "right_elbow",
                        "left_wrist",
                        "right_wrist",
                    ]
                ]
                ax = pd.DataFrame(columns=a.columns)
                ay = pd.DataFrame(columns=a.columns)

                for i in range(len(a)):
                    info = a.loc[i, :]
                    xlst = []
                    ylst = []
                    for j in info:
                        x, y = j
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

                info = pd.DataFrame(
                    columns=[
                        "eye-eye",
                        "ear-ear",
                        "shoulder-shoulder",
                        "nose-mid_shoulder",
                        "eye-mid_shoulder",
                        "right_hand-yes",
                        "left_hand-yes",
                        "hand",
                    ]
                )
                for i in range(len(a)):
                    bx = ax.loc[i, :]
                    by = ay.loc[i, :]
                    lst = []
                    lst.append(
                        (by["right_eye"] - by["left_eye"])
                        / (bx["right_eye"] - bx["left_eye"])
                    )
                    lst.append(
                        (by["right_ear"] - by["left_ear"])
                        / (bx["right_ear"] - bx["left_ear"])
                    )
                    lst.append(
                        (by["right_shoulder"] - by["left_shoulder"])
                        / (bx["right_shoulder"] - bx["left_shoulder"])
                    )
                    lst.append(
                        (by["nose"] - by["mid_shoulder"])
                        / max((bx["nose"] - bx["mid_shoulder"]), 1e-6)
                    )
                    lst.append(
                        ((by["right_eye"] + by["left_eye"]) / 2 - by["mid_shoulder"])
                        / max(
                            (
                                (bx["right_eye"] + bx["left_eye"]) / 2
                                - bx["mid_shoulder"]
                            ),
                            1e-6,
                        )
                    )
                    right_hand = (bx["right_wrist"] != -1) and (bx["right_elbow"] != -1)
                    left_hand = (bx["left_wrist"] != -1) and (bx["left_elbow"] != -1)
                    lst.append(right_hand)
                    lst.append(left_hand)
                    if right_hand and left_hand:
                        lst.append("both")
                    elif right_hand and not left_hand:
                        lst.append("right")
                    elif left_hand and not right_hand:
                        lst.append("left")
                    else:
                        lst.append("none")
                    info.loc[i, :] = lst
                info["seconds"] = pose_sec

                vertical_threshold = np.tan(pose_vertical_threshold)
                horizontal_threshold = np.tan(pose_horizontal_threshold)
                info_ = pd.DataFrame(
                    columns=[
                        "face_align",
                        "body_align",
                        "vertical_align",
                        "hand",
                        "seconds",
                    ]
                )
                for i in range(len(info)):
                    lst = []
                    (
                        eye_eye,
                        ear_ear,
                        shd_shd,
                        nose_chest,
                        eye_chest,
                        rhand,
                        lhand,
                        hand,
                        secs,
                    ) = info.loc[i, :]
                    # 얼굴 align
                    if (
                        abs(eye_eye) < horizontal_threshold
                        or abs(ear_ear) < horizontal_threshold
                    ):
                        lst.append(True)
                    else:
                        lst.append(False)
                    # 몸통 align
                    if abs(shd_shd) < horizontal_threshold:
                        lst.append(True)
                    else:
                        lst.append(False)
                    # 얼굴-몸통 삐딱
                    if (
                        abs(nose_chest) > vertical_threshold
                        or abs(eye_chest) > vertical_threshold
                    ):
                        lst.append(True)
                    else:
                        lst.append(False)
                    # 손 출현
                    if rhand or lhand:
                        lst.append(True)
                    else:
                        lst.append(False)
                    lst.append(secs)
                    info_.loc[i, :] = lst

                with st.expander("More Information - Pose Estimation Result"):
                    st.subheader("Pose result")
                    st.dataframe(pose_result)
                    st.subheader("Pose Angle")
                    st.dataframe(info)
                    st.subheader("Pose is Align?")
                    st.dataframe(info_)

                x = np.linspace(0, len(result), 200)

                st.subheader("구간 별 동영상을 확인해보세요")
                tab1_, tab2_, tab3_, tab4_ = st.tabs(
                    ["Face Align", "Body Align", "Vertical Align", "Hand"]
                )
                pose1, pose2, pose3, pose4 = [], [], [], []
                if len(st.session_state.pose_time) > 0:
                    for idx, (start_sec, end_sec, start, end, type) in enumerate(
                        st.session_state.pose_time
                    ):
                        if type == "face":
                            pose1.append([idx, start_sec, end_sec, start, end])
                        elif type == "shoulder":
                            pose2.append([idx, start_sec, end_sec, start, end])
                        elif type == "body":
                            pose3.append([idx, start_sec, end_sec, start, end])
                        elif type == "hand":
                            pose4.append([idx, start_sec, end_sec, start, end])
                else:
                    pass

                with tab1_:
                    if len(pose1) > 0:
                        for idx, start_sec, end_sec, start, end in pose1:
                            with st.expander(
                                f"🔴 {round(start_sec, 2)}초 ~ {round(end_sec, 2)}초의 고개가 기울어졌습니다."
                            ):
                                col1, col2 = st.columns(2)
                                with col1:
                                    mode = st.selectbox(
                                        "What kind of line chart do you want?",
                                        ("Eye-Eye", "Ear-Ear", "Horizontal"),
                                        key=200 + idx,
                                    )
                                    fig, ax = plt.subplots()
                                    ax.set_xlabel("Time(sec)")
                                    ax.set_ylabel("Angle")
                                    ax.set_xticks(
                                        [i for idx, i in enumerate(x) if idx % 15 == 1]
                                    )
                                    ax.set_xticklabels(
                                        [
                                            round(i / 30, 1)
                                            for idx, i in enumerate(x)
                                            if idx % 15 == 1
                                        ]
                                    )
                                    ax.tick_params(axis="x", rotation=30)
                                    if mode == "Eye-Eye":
                                        angle_y = [
                                            i * (180 / math.pi)
                                            for i in np.arctan(
                                                info["eye-eye"].astype(np.float64)
                                            )
                                        ]
                                        ax.axhline(
                                            y=pose_horizontal_threshold
                                            * (180 / math.pi),
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.axhline(
                                            y=-pose_horizontal_threshold
                                            * (180 / math.pi),
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.axhline(
                                            y=0,
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.plot(angle_y, color="skyblue")

                                    elif mode == "Ear-Ear":
                                        angle_y = [
                                            i * (180 / math.pi)
                                            for i in np.arctan(
                                                info["ear-ear"].astype(np.float64)
                                            )
                                        ]
                                        ax.axhline(
                                            y=pose_horizontal_threshold
                                            * (180 / math.pi),
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.axhline(
                                            y=-pose_horizontal_threshold
                                            * (180 / math.pi),
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.axhline(
                                            y=0,
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.plot(angle_y, color="skyblue")

                                    elif mode == "Horizontal":
                                        angle_y1 = [
                                            i * (180 / math.pi)
                                            for i in np.arctan(
                                                info["eye-eye"].astype(np.float64)
                                            )
                                        ]
                                        angle_y2 = [
                                            i * (180 / math.pi)
                                            for i in np.arctan(
                                                info["ear-ear"].astype(np.float64)
                                            )
                                        ]
                                        angle_y3 = [
                                            i * (180 / math.pi)
                                            for i in np.arctan(
                                                info["shoulder-shoulder"].astype(
                                                    np.float64
                                                )
                                            )
                                        ]
                                        ax.axhline(
                                            y=pose_horizontal_threshold
                                            * (180 / math.pi),
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.axhline(
                                            y=-pose_horizontal_threshold
                                            * (180 / math.pi),
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.axhline(
                                            y=0,
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.plot(
                                            angle_y1, color="skyblue", label="Eye-Eye"
                                        )
                                        ax.plot(
                                            angle_y2,
                                            color="yellowgreen",
                                            label="Ear-Ear",
                                        )
                                        ax.plot(
                                            angle_y3,
                                            color="khaki",
                                            label="Shoulder-Shoulder",
                                        )
                                        ax.legend(loc="best")

                                    ax.axvline(
                                        x=start_sec * 30,
                                        linestyle="--",
                                        color="black",
                                        alpha=0.5,
                                    )
                                    ax.axvline(
                                        x=end_sec * 30,
                                        linestyle="--",
                                        color="black",
                                        alpha=0.5,
                                    )

                                    st.pyplot(fig)

                                with col2:
                                    st_show_video(
                                        "/".join(
                                            [
                                                st.session_state.result_dir,
                                                "slice",
                                                f"pose_slice_{idx}.webm",
                                            ]
                                        )
                                    )
                                    st.session_state.is_okay[
                                        f"pose_{idx}_{start}_{end}"
                                    ] = st.checkbox(
                                        "👌 이 분석 결과에 만족하지 않으신가요?", key=idx + 300
                                    )
                                    if st.session_state.is_okay[
                                        f"pose_{idx}_{start}_{end}"
                                    ]:
                                        st.success("감사합니다.", icon="🔥")

                    else:
                        with st.expander("🟢 얼굴이 잘 정렬되어 있습니다."):
                            col1, col2 = st.columns(2)
                            with col1:
                                mode = st.selectbox(
                                    "What kind of line chart do you want?",
                                    ("Eye-Eye", "Ear-Ear", "Horizontal"),
                                    key=20000,
                                )
                                fig, ax = plt.subplots()
                                ax.set_xlabel("Time(sec)")
                                ax.set_ylabel("Angle")
                                ax.set_xticks(
                                    [i for idx, i in enumerate(x) if idx % 15 == 1]
                                )
                                ax.set_xticklabels(
                                    [
                                        round(i / 30, 1)
                                        for idx, i in enumerate(x)
                                        if idx % 15 == 1
                                    ]
                                )
                                ax.tick_params(axis="x", rotation=30)
                                if mode == "Eye-Eye":
                                    angle_y = [
                                        i * (180 / math.pi)
                                        for i in np.arctan(
                                            info["eye-eye"].astype(np.float64)
                                        )
                                    ]
                                    ax.axhline(
                                        y=pose_horizontal_threshold * (180 / math.pi),
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.axhline(
                                        y=-pose_horizontal_threshold * (180 / math.pi),
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.axhline(
                                        y=0,
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.plot(angle_y, color="skyblue")

                                elif mode == "Ear-Ear":
                                    angle_y = [
                                        i * (180 / math.pi)
                                        for i in np.arctan(
                                            info["ear-ear"].astype(np.float64)
                                        )
                                    ]
                                    ax.axhline(
                                        y=pose_horizontal_threshold * (180 / math.pi),
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.axhline(
                                        y=-pose_horizontal_threshold * (180 / math.pi),
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.axhline(
                                        y=0,
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.plot(angle_y, color="skyblue")

                                elif mode == "Horizontal":
                                    angle_y1 = [
                                        i * (180 / math.pi)
                                        for i in np.arctan(
                                            info["eye-eye"].astype(np.float64)
                                        )
                                    ]
                                    angle_y2 = [
                                        i * (180 / math.pi)
                                        for i in np.arctan(
                                            info["ear-ear"].astype(np.float64)
                                        )
                                    ]
                                    angle_y3 = [
                                        i * (180 / math.pi)
                                        for i in np.arctan(
                                            info["shoulder-shoulder"].astype(np.float64)
                                        )
                                    ]
                                    ax.axhline(
                                        y=pose_horizontal_threshold * (180 / math.pi),
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.axhline(
                                        y=-pose_horizontal_threshold * (180 / math.pi),
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.axhline(
                                        y=0,
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.plot(angle_y1, color="skyblue", label="Eye-Eye")
                                    ax.plot(
                                        angle_y2, color="yellowgreen", label="Ear-Ear"
                                    )
                                    ax.plot(
                                        angle_y3,
                                        color="khaki",
                                        label="Shoulder-Shoulder",
                                    )
                                    ax.legend(loc="best")

                                st.pyplot(fig)

                            with col2:
                                st.session_state.is_okay["pose_face_all"] = st.checkbox(
                                    "👌 이 분석 결과에 만족하지 않으신가요?", key=200000
                                )
                                if st.session_state.is_okay["pose_face_all"]:
                                    st.success("감사합니다.", icon="🔥")

                with tab2_:
                    if len(pose2) > 0:
                        for idx, start_sec, end_sec, start, end in pose2:
                            with st.expander(
                                f"🔴 {round(start_sec, 2)}초 ~ {round(end_sec, 2)}초의 어깨선이 기울어졌습니다."
                            ):
                                col1, col2 = st.columns(2)
                                with col1:
                                    fig, ax = plt.subplots()
                                    ax.set_xlabel("Time(sec)")
                                    ax.set_ylabel("Angle")
                                    ax.set_xticks(
                                        [i for idx, i in enumerate(x) if idx % 15 == 1]
                                    )
                                    ax.set_xticklabels(
                                        [
                                            round(i / 30, 1)
                                            for idx, i in enumerate(x)
                                            if idx % 15 == 1
                                        ]
                                    )
                                    ax.tick_params(axis="x", rotation=30)

                                    mode = st.selectbox(
                                        "What kind of line chart do you want?",
                                        ("Shoulder-Shoulder", "Horizontal"),
                                        key=500 + idx,
                                    )

                                    if mode == "Shoulder-Shoulder":
                                        angle_y = [
                                            i * (180 / math.pi)
                                            for i in np.arctan(
                                                info["shoulder-shoulder"].astype(
                                                    np.float64
                                                )
                                            )
                                        ]
                                        ax.axhline(
                                            y=pose_horizontal_threshold
                                            * (180 / math.pi),
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.axhline(
                                            y=-pose_horizontal_threshold
                                            * (180 / math.pi),
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.axhline(
                                            y=0,
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.plot(angle_y, color="skyblue")

                                    elif mode == "Horizontal":
                                        angle_y1 = [
                                            i * (180 / math.pi)
                                            for i in np.arctan(
                                                info["eye-eye"].astype(np.float64)
                                            )
                                        ]
                                        angle_y2 = [
                                            i * (180 / math.pi)
                                            for i in np.arctan(
                                                info["ear-ear"].astype(np.float64)
                                            )
                                        ]
                                        angle_y3 = [
                                            i * (180 / math.pi)
                                            for i in np.arctan(
                                                info["shoulder-shoulder"].astype(
                                                    np.float64
                                                )
                                            )
                                        ]
                                        ax.axhline(
                                            y=pose_horizontal_threshold
                                            * (180 / math.pi),
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.axhline(
                                            y=-pose_horizontal_threshold
                                            * (180 / math.pi),
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.axhline(
                                            y=0,
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.plot(
                                            angle_y1, color="skyblue", label="Eye-Eye"
                                        )
                                        ax.plot(
                                            angle_y2,
                                            color="yellowgreen",
                                            label="Ear-Ear",
                                        )
                                        ax.plot(
                                            angle_y3,
                                            color="khaki",
                                            label="Shoulder-Shoulder",
                                        )
                                        ax.legend(loc="best")

                                    ax.axvline(
                                        x=start_sec * 30,
                                        linestyle="--",
                                        color="black",
                                        alpha=0.5,
                                    )
                                    ax.axvline(
                                        x=end_sec * 30,
                                        linestyle="--",
                                        color="black",
                                        alpha=0.5,
                                    )

                                    st.pyplot(fig)

                                with col2:
                                    st_show_video(
                                        "/".join(
                                            [
                                                st.session_state.result_dir,
                                                "slice",
                                                f"pose_slice_{idx}.webm",
                                            ]
                                        )
                                    )
                                    st.session_state.is_okay[
                                        f"pose_{idx}_{start}_{end}"
                                    ] = st.checkbox(
                                        "👌 이 분석 결과에 만족하지 않으신가요?", key=idx + 600
                                    )
                                    if st.session_state.is_okay[
                                        f"pose_{idx}_{start}_{end}"
                                    ]:
                                        st.success("감사합니다.", icon="🔥")

                    else:
                        with st.expander("🟢 어깨선이 잘 정렬되어 있습니다."):
                            col1, col2 = st.columns(2)
                            with col1:
                                fig, ax = plt.subplots()
                                ax.set_xlabel("Time(sec)")
                                ax.set_ylabel("Angle")
                                ax.set_xticks(
                                    [i for idx, i in enumerate(x) if idx % 15 == 1]
                                )
                                ax.set_xticklabels(
                                    [
                                        round(i / 30, 1)
                                        for idx, i in enumerate(x)
                                        if idx % 15 == 1
                                    ]
                                )
                                ax.tick_params(axis="x", rotation=30)

                                mode = st.selectbox(
                                    "What kind of line chart do you want?",
                                    ("Shoulder-Shoulder", "Horizontal"),
                                    key=30000,
                                )

                                if mode == "Shoulder-Shoulder":
                                    angle_y = [
                                        i * (180 / math.pi)
                                        for i in np.arctan(
                                            info["shoulder-shoulder"].astype(np.float64)
                                        )
                                    ]
                                    ax.axhline(
                                        y=pose_horizontal_threshold * (180 / math.pi),
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.axhline(
                                        y=-pose_horizontal_threshold * (180 / math.pi),
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.axhline(
                                        y=0,
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.plot(angle_y, color="skyblue")

                                elif mode == "Horizontal":
                                    angle_y1 = [
                                        i * (180 / math.pi)
                                        for i in np.arctan(
                                            info["eye-eye"].astype(np.float64)
                                        )
                                    ]
                                    angle_y2 = [
                                        i * (180 / math.pi)
                                        for i in np.arctan(
                                            info["ear-ear"].astype(np.float64)
                                        )
                                    ]
                                    angle_y3 = [
                                        i * (180 / math.pi)
                                        for i in np.arctan(
                                            info["shoulder-shoulder"].astype(np.float64)
                                        )
                                    ]
                                    ax.axhline(
                                        y=pose_horizontal_threshold * (180 / math.pi),
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.axhline(
                                        y=-pose_horizontal_threshold * (180 / math.pi),
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.axhline(
                                        y=0,
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.plot(angle_y1, color="skyblue", label="Eye-Eye")
                                    ax.plot(
                                        angle_y2, color="yellowgreen", label="Ear-Ear"
                                    )
                                    ax.plot(
                                        angle_y3,
                                        color="khaki",
                                        label="Shoulder-Shoulder",
                                    )
                                    ax.legend(loc="best")

                                st.pyplot(fig)

                            with col2:
                                st.session_state.is_okay[
                                    "pose_shoulder_all"
                                ] = st.checkbox("👌 이 분석 결과에 만족하지 않으신가요?", key=300000)
                                if st.session_state.is_okay["pose_shoulder_all"]:
                                    st.success("감사합니다.", icon="🔥")

                with tab3_:
                    if len(pose3) > 0:
                        for idx, start_sec, end_sec, start, end in pose3:
                            with st.expander(
                                f"🔴 {round(start_sec, 2)}초 ~ {round(end_sec, 2)}초의 몸이 기울어졌습니다."
                            ):
                                col1, col2 = st.columns(2)
                                with col1:
                                    fig, ax = plt.subplots()
                                    ax.set_xlabel("Time(sec)")
                                    ax.set_ylabel("Angle")
                                    ax.set_xticks(
                                        [i for idx, i in enumerate(x) if idx % 15 == 1]
                                    )
                                    ax.set_xticklabels(
                                        [
                                            round(i / 30, 1)
                                            for idx, i in enumerate(x)
                                            if idx % 15 == 1
                                        ]
                                    )
                                    ax.tick_params(axis="x", rotation=30)

                                    mode = st.selectbox(
                                        "What kind of line chart do you want?",
                                        (
                                            "Nose-Mid Shoulder",
                                            "Eye-Mid Shoulder",
                                            "Vertical",
                                        ),
                                        key=800 + idx,
                                    )

                                    if mode == "Nose-Mid Shoulder":
                                        angle_y = [
                                            i * (180 / math.pi)
                                            for i in np.arctan(
                                                info["nose-mid_shoulder"].astype(
                                                    np.float64
                                                )
                                            )
                                        ]
                                        ax.axhline(
                                            y=pose_vertical_threshold * (180 / math.pi),
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.axhline(
                                            y=180
                                            - pose_vertical_threshold * (180 / math.pi),
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.axhline(
                                            y=90,
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.plot(
                                            [i if i > 0 else i + 180 for i in angle_y],
                                            color="skyblue",
                                        )
                                        ax.set_ylim(70, 110)

                                    elif mode == "Eye-Mid Shoulder":
                                        angle_y = [
                                            i * (180 / math.pi)
                                            for i in np.arctan(
                                                info["eye-mid_shoulder"].astype(
                                                    np.float64
                                                )
                                            )
                                        ]
                                        ax.axhline(
                                            y=pose_vertical_threshold * (180 / math.pi),
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.axhline(
                                            y=180
                                            - pose_vertical_threshold * (180 / math.pi),
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.axhline(
                                            y=90,
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.plot(
                                            [i if i > 0 else i + 180 for i in angle_y],
                                            color="skyblue",
                                        )
                                        ax.set_ylim(70, 110)

                                    elif mode == "Vertical":
                                        angle_y1 = [
                                            i * (180 / math.pi)
                                            for i in np.arctan(
                                                info["nose-mid_shoulder"].astype(
                                                    np.float64
                                                )
                                            )
                                        ]
                                        angle_y2 = [
                                            i * (180 / math.pi)
                                            for i in np.arctan(
                                                info["eye-mid_shoulder"].astype(
                                                    np.float64
                                                )
                                            )
                                        ]
                                        ax.axhline(
                                            y=pose_vertical_threshold * (180 / math.pi),
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.axhline(
                                            y=180
                                            - pose_vertical_threshold * (180 / math.pi),
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.axhline(
                                            y=90,
                                            color="lightcoral",
                                            linestyle="--",
                                            alpha=0.5,
                                        )
                                        ax.plot(
                                            [i if i > 0 else i + 180 for i in angle_y1],
                                            color="skyblue",
                                            label="Nose-Mid Shoulder",
                                        )
                                        ax.plot(
                                            [i if i > 0 else i + 180 for i in angle_y2],
                                            color="yellowgreen",
                                            label="Eye-Mid Shoulder",
                                        )
                                        ax.set_ylim(70, 110)
                                        ax.legend(loc="best")

                                    ax.axvline(
                                        x=start_sec * 30,
                                        linestyle="--",
                                        color="black",
                                        alpha=0.5,
                                    )
                                    ax.axvline(
                                        x=end_sec * 30,
                                        linestyle="--",
                                        color="black",
                                        alpha=0.5,
                                    )

                                    st.pyplot(fig)
                                with col2:
                                    st_show_video(
                                        "/".join(
                                            [
                                                st.session_state.result_dir,
                                                "slice",
                                                f"pose_slice_{idx}.webm",
                                            ]
                                        )
                                    )
                                    st.session_state.is_okay[
                                        f"pose_{idx}_{start}_{end}"
                                    ] = st.checkbox(
                                        "👌 이 분석 결과에 만족하지 않으신가요?", key=idx + 900
                                    )
                                    if st.session_state.is_okay[
                                        f"pose_{idx}_{start}_{end}"
                                    ]:
                                        st.success("감사합니다.", icon="🔥")
                    else:
                        with st.expander("🟢 몸과 얼굴이 잘 정렬되어 있습니다."):
                            col1, col2 = st.columns(2)
                            with col1:
                                fig, ax = plt.subplots()
                                ax.set_xlabel("Time(sec)")
                                ax.set_ylabel("Angle")
                                ax.set_xticks(
                                    [i for idx, i in enumerate(x) if idx % 15 == 1]
                                )
                                ax.set_xticklabels(
                                    [
                                        round(i / 30, 1)
                                        for idx, i in enumerate(x)
                                        if idx % 15 == 1
                                    ]
                                )
                                ax.tick_params(axis="x", rotation=30)

                                mode = st.selectbox(
                                    "What kind of line chart do you want?",
                                    (
                                        "Nose-Mid Shoulder",
                                        "Eye-Mid Shoulder",
                                        "Vertical",
                                    ),
                                    key=40000,
                                )

                                if mode == "Nose-Mid Shoulder":
                                    angle_y = [
                                        i * (180 / math.pi)
                                        for i in np.arctan(
                                            info["nose-mid_shoulder"].astype(np.float64)
                                        )
                                    ]
                                    ax.axhline(
                                        y=pose_vertical_threshold * (180 / math.pi),
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.axhline(
                                        y=180
                                        - pose_vertical_threshold * (180 / math.pi),
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.axhline(
                                        y=90,
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.plot(
                                        [i if i > 0 else i + 180 for i in angle_y],
                                        color="skyblue",
                                    )
                                    ax.set_ylim(70, 110)

                                elif mode == "Eye-Mid Shoulder":
                                    angle_y = [
                                        i * (180 / math.pi)
                                        for i in np.arctan(
                                            info["eye-mid_shoulder"].astype(np.float64)
                                        )
                                    ]
                                    ax.axhline(
                                        y=pose_vertical_threshold * (180 / math.pi),
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.axhline(
                                        y=180
                                        - pose_vertical_threshold * (180 / math.pi),
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.axhline(
                                        y=90,
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.plot(
                                        [i if i > 0 else i + 180 for i in angle_y],
                                        color="skyblue",
                                    )
                                    ax.set_ylim(70, 110)

                                elif mode == "Vertical":
                                    angle_y1 = [
                                        i * (180 / math.pi)
                                        for i in np.arctan(
                                            info["nose-mid_shoulder"].astype(np.float64)
                                        )
                                    ]
                                    angle_y2 = [
                                        i * (180 / math.pi)
                                        for i in np.arctan(
                                            info["eye-mid_shoulder"].astype(np.float64)
                                        )
                                    ]
                                    ax.axhline(
                                        y=pose_vertical_threshold * (180 / math.pi),
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.axhline(
                                        y=180
                                        - pose_vertical_threshold * (180 / math.pi),
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.axhline(
                                        y=90,
                                        color="lightcoral",
                                        linestyle="--",
                                        alpha=0.5,
                                    )
                                    ax.plot(
                                        [i if i > 0 else i + 180 for i in angle_y1],
                                        color="skyblue",
                                        label="Nose-Mid Shoulder",
                                    )
                                    ax.plot(
                                        [i if i > 0 else i + 180 for i in angle_y2],
                                        color="yellowgreen",
                                        label="Eye-Mid Shoulder",
                                    )
                                    ax.set_ylim(70, 110)
                                    ax.legend(loc="best")

                                st.pyplot(fig)

                            with col2:
                                st.session_state.is_okay["pose_body_all"] = st.checkbox(
                                    "👌 이 분석 결과에 만족하지 않으신가요?", key=400000
                                )
                                if st.session_state.is_okay["pose_body_all"]:
                                    st.success("감사합니다", icon="🔥")

                with tab4_:
                    if len(pose4) > 0:
                        for idx, start_sec, end_sec, start, end in pose4:
                            with st.expander(
                                f"🔴 {round(start_sec, 2)}초 ~ {round(end_sec, 2)}초에 손이 나왔습니다."
                            ):
                                col1, col2 = st.columns(2)
                                with col1:
                                    numhand = info.hand.replace(
                                        ["none", "left", "right", "both"], [0, 1, 2, 3]
                                    )

                                    fig, ax = plt.subplots()
                                    ax.set_xlabel("Time(sec)")
                                    ax.set_ylabel("Hand or No")

                                    ax.set_xticks(
                                        [i for idx, i in enumerate(x) if idx % 15 == 1]
                                    )
                                    ax.set_xticklabels(
                                        [
                                            round(i / 30, 1)
                                            for idx, i in enumerate(x)
                                            if idx % 15 == 1
                                        ]
                                    )
                                    ax.set_yticks([0, 1, 2, 3])
                                    ax.set_yticklabels(
                                        [
                                            "No Hand",
                                            "Left Hand",
                                            "Right Hand",
                                            "Both Hand",
                                        ]
                                    )
                                    ax.tick_params(axis="x", rotation=30)
                                    ax.plot(
                                        numhand,
                                        color="skyblue",
                                        label="Nose-Mid Shoulder",
                                    )

                                    ax.axvline(
                                        x=start_sec * 30,
                                        linestyle="--",
                                        color="black",
                                        alpha=0.5,
                                    )
                                    ax.axvline(
                                        x=end_sec * 30,
                                        linestyle="--",
                                        color="black",
                                        alpha=0.5,
                                    )

                                    st.pyplot(fig)

                                with col2:
                                    st_show_video(
                                        "/".join(
                                            [
                                                st.session_state.result_dir,
                                                "slice",
                                                f"pose_slice_{idx}.webm",
                                            ]
                                        )
                                    )
                                    st.session_state.is_okay[
                                        f"pose_{idx}_{start}_{end}"
                                    ] = st.checkbox(
                                        "👌 이 분석 결과에 만족하지 않으신가요?", key=idx + 1100
                                    )
                                    if st.session_state.is_okay[
                                        f"pose_{idx}_{start}_{end}"
                                    ]:
                                        st.success("감사합니다.", icon="🔥")
                    else:
                        with st.expander("🟢 손이 나오지 않았습니다."):
                            col1, col2 = st.columns(2)
                            with col1:
                                numhand = info.hand.replace(
                                    ["none", "left", "right", "both"], [0, 1, 2, 3]
                                )

                                fig, ax = plt.subplots()
                                ax.set_xlabel("Time(sec)")
                                ax.set_ylabel("Hand or No")

                                ax.set_xticks(
                                    [i for idx, i in enumerate(x) if idx % 15 == 1]
                                )
                                ax.set_xticklabels(
                                    [
                                        round(i / 30, 1)
                                        for idx, i in enumerate(x)
                                        if idx % 15 == 1
                                    ]
                                )
                                ax.set_yticks([0, 1, 2, 3])
                                ax.set_yticklabels(
                                    ["No Hand", "Left Hand", "Right Hand", "Both Hand"]
                                )
                                ax.tick_params(axis="x", rotation=30)
                                ax.plot(
                                    numhand, color="skyblue", label="Nose-Mid Shoulder"
                                )
                                st.pyplot(fig)

                            with col2:
                                st.session_state.is_okay["pose_hand_all"] = st.checkbox(
                                    "👌 이 분석 결과에 만족하지 않으신가요?", key=500000
                                )
                                if st.session_state.is_okay["pose_hand_all"]:
                                    st.success("감사합니다", icon="🔥")

            with tab3:
                st.header("Eye")
                eye_video = cv2.VideoCapture(
                    f"./{VIDEO_PATH.split('/')[1]}/{VIDEO_PATH.split('/')[2]}/eye_recording.webm"
                )
                eye_video_len = eye_video.get(cv2.CAP_PROP_FRAME_COUNT) / max(
                    eye_video.get(cv2.CAP_PROP_FPS), 1e-6
                )
                eye_sec = [
                    eye_video_len / len(eye_result) * (i + 1)
                    for i in range(len(eye_result))
                ]
                eye_result["seconds"] = eye_sec

                with st.expander("More Information - Eye Tracking Result"):
                    st.dataframe(eye_result)

                x = np.linspace(0, len(eye_result), 200)

                st.subheader("구간 별 동영상을 확인해보세요")
                if len(st.session_state.eye_time) > 0:
                    for idx, (start_sec, end_sec, start, end, direction) in enumerate(
                        sorted(st.session_state.eye_time)
                    ):
                        with st.expander(
                            f"🔴 {round(start_sec, 2)}초 ~ {round(end_sec, 2)}초의 시선이 {direction}을 응시하고 있습니다."
                        ):
                            col1, col2 = st.columns(2)
                            with col1:
                                fig, ax = plt.subplots()
                                ax.set_xlabel("Time(sec)")
                                ax.set_ylabel("Emotion")

                                ax.set_xticks(
                                    [i for idx, i in enumerate(x) if idx % 15 == 1]
                                )
                                ax.set_xticklabels(
                                    [
                                        round(i / 30, 1)
                                        for idx, i in enumerate(x)
                                        if idx % 15 == 1
                                    ]
                                )
                                ax.tick_params(axis="x", rotation=30)

                                numeye = eye_result.tracking.replace(
                                    ["Right", "Center", "Left"], [-1, 0, 1]
                                )

                                model_eye = make_interp_spline(
                                    [i for i in range(len(eye_result))], numeye
                                )

                                interpol_eye = model_eye(x)

                                ax.plot(
                                    x, interpol_eye, color="skyblue", label="emotion"
                                )
                                ax.set_ylim(-1.3, 1.3)
                                ax.set_yticks([-1, 0, 1])
                                ax.set_yticklabels(["Right", "Center", "Left"])
                                ax.axvline(
                                    x=start_sec * 30,
                                    color="black",
                                    linestyle="--",
                                    alpha=0.5,
                                )
                                ax.axvline(
                                    x=end_sec * 30,
                                    color="black",
                                    linestyle="--",
                                    alpha=0.5,
                                )
                                ax.axhline(
                                    y=0, color="lightcoral", linestyle="--", alpha=0.5
                                )
                                st.pyplot(fig)

                            with col2:
                                st_show_video(
                                    "/".join(
                                        [
                                            st.session_state.result_dir,
                                            "slice",
                                            f"eye_slice_{idx}.webm",
                                        ]
                                    )
                                )
                                st.session_state.is_okay[
                                    f"eye_{idx}_{start}_{end}"
                                ] = st.checkbox(
                                    "👌 이 분석 결과에 만족하지 않으신가요?", key=idx + 1300
                                )
                                if st.session_state.is_okay[f"eye_{idx}_{start}_{end}"]:
                                    st.success("감사합니다.", icon="🔥")
                else:
                    with st.expander("🟢 정면을 잘 응시하고 있습니다."):
                        col1, col2 = st.columns(2)
                        with col1:
                            fig, ax = plt.subplots()
                            ax.set_xlabel("Time(sec)")
                            ax.set_ylabel("Emotion")

                            ax.set_xticks(
                                [i for idx, i in enumerate(x) if idx % 15 == 1]
                            )
                            ax.set_xticklabels(
                                [
                                    round(i / 30, 1)
                                    for idx, i in enumerate(x)
                                    if idx % 15 == 1
                                ]
                            )
                            ax.tick_params(axis="x", rotation=30)

                            numeye = eye_result.tracking.replace(
                                ["Right", "Center", "Left"], [-1, 0, 1]
                            )

                            model_eye = make_interp_spline(
                                [i for i in range(len(eye_result))], numeye
                            )

                            interpol_eye = model_eye(x)

                            ax.plot(x, interpol_eye, color="skyblue", label="emotion")
                            ax.set_ylim(-1.3, 1.3)
                            ax.set_yticks([-1, 0, 1])
                            ax.set_yticklabels(["Right", "Center", "Left"])
                            ax.axhline(
                                y=0, color="lightcoral", linestyle="--", alpha=0.5
                            )
                            st.pyplot(fig)

                        with col2:
                            st.session_state.is_okay["eye_all"] = st.checkbox(
                                "👌 이 분석 결과에 만족하지 않으신가요?", key=600000
                            )
                            if st.session_state.is_okay["eye_all"]:
                                st.success("감사합니다.", icon="🔥")

            send_feedback = st.button("🧡 Send Feedback to Developer 💚")
            if send_feedback:
                st.write("모든 분석 결과에 대해 피드백을 하셨는지 다시 확인해주세요!")
                st.write(st.session_state.is_okay)
                send = st.button("Send")
                if send:
                    feedbackDB = FeedbackDB(
                        path=st.session_state.result_dir, data=st.session_state.is_okay
                    )
                    feedbackDB.save_data()

    else:
        st.subheader("면접 영상이 제대로 저장되지 않았습니다. 다시 면접 영상을 녹화해주세요.")
else:
    st.subheader("면접 영상이 선택되지 않았습니다. 다시 면접 영상을 녹화해주세요.")
