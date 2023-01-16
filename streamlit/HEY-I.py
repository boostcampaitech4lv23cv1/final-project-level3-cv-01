import streamlit as st

video_file = open('./db/output_230116_112926.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)