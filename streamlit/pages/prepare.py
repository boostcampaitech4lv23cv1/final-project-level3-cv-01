import sys
import cv2
import numpy as np
import streamlit as st

st.set_page_config(layout="wide")

# if not "name" in st.session_state.keys():
#     st.warning("HEY-I 페이지에서 이름과 번호를 입력하세요")
#     st.stop()

st.markdown("# 면접 자세를 확인하세요")
st.markdown("## 어깨와 얼굴을 선에 맞춰주세요")

col1, col2 = st.columns(2)

with col2:
    st.markdown(
        """ 
        ### 올바른 면접 자세란?
        #### ✔ 1. 허리를 피세요
        #### ✔ 2. 어깨를 피세요
        #### ✔ 3. 눈은 카메라를 응시하세요
        #### ✔ 4. 옷 매무새와 머리를 정리하세요
        #### ❌ 1. 고개를 흔들지 마세요
        #### ❌ 2. 너무 많은 손동작은 어수선해 보일 수 있어요
        #### ❌ 3. 너무 딱딱한 표정은 어색해 보일 수 있어요
        """
    )

with col1:
    
    st.warning('현재 webrtc 와 http 호환성 문제로 인해 카메라를 켤 수 없습니다.', icon="🚨")

    # stframe = st.empty()

    # with st.spinner("Get ready for camera"):

    #     cap = cv2.VideoCapture(0)

    #     if not cap.isOpened():
    #         print("Camera open failed!")
    #         st.write("Camera open failed!")
    #         sys.exit()

    #     w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     fps = cap.get(cv2.CAP_PROP_FPS)

    #     person = np.array(cv2.resize(cv2.imread("streamlit/person.png"), (w, h))) / 255
    #     person = np.array(person, dtype="u1")

    # while True:
    #     ret, frame = cap.read()

    #     if not ret:
    #         break

    #     stframe.image(frame * person, channels="BGR")
    person = np.array(cv2.imread("streamlit/person.png"))
    person = np.array(person, dtype="u1")
    st.image(person)
