import os
import sys
import streamlit as st

sys.path.append(os.getcwd())
import streamlit as st


# Basic App Scaffolding
st.title("HEY-I")

st.subheader("이 사이트는 inference 결과를 미리 볼 수 있는 사이트입니다.")
st.warning("이 페이지에서는 영상을 녹화하거나 업로드를 하여 원하는 영상을 분석할 수 있지만 이 사이트에서는 준비된 영상의 결과만을 보여줍니다!", icon="🚨")
st.markdown("**result page** 에서 결과를 확인하세요")

video_file = open('./deploy/result/recording.webm', "rb")
video_bytes = video_file.read()
with st.expander("분석 영상을 확인하려면 이곳에서 확인하세요"):
    st.video(video_bytes)