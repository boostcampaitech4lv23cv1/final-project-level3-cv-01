import cv2
import numpy as np
import streamlit as st

st.set_page_config(layout="wide")

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

    person = np.array(cv2.imread("streamlit/person.png"))
    person = np.array(person, dtype="u1")
    st.image(person)
