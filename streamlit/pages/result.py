import streamlit as st
import sys
import os

st.title('HEY-I')
st.subheader('면접 영상 분석 결과입니다.')

if 'video_dir' in st.session_state.keys():
    if os.path.exists(st.session_state.video_dir):
        video_file = open(st.session_state.video_dir, 'rb')
        video_bytes = video_file.read()
        st.write('선택된 영상입니다.')    
        st.video(video_bytes)




