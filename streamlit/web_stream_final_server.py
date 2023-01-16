import os
import cv2
import sys
import time
import tempfile
from pytz import timezone
from datetime import datetime
import streamlit as st
import mediapipe as mp

# Basic App Scaffolding
st.title('HEY-I')

# Create Sidebar
st.sidebar.title('Settings')

## Get Video
temp_file = tempfile.NamedTemporaryFile(delete=False)

stframe = st.empty()
number = st.sidebar.number_input('분 입력',1,10)
start_recording = st.sidebar.button('Start Recordinging')
end_recording = st.sidebar.button('End Recordinging')

if start_recording and not end_recording:
    video = cv2.VideoCapture(0)

    #Load Web Camera
    if not (video.isOpened()):
        print("File isn't opend!!")

    #Set Video File Property
    w = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framecount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    #video.set(cv2.CAP_PROP_FPS, 10) # fps 설정
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    delay = 6
    print('fps:',fps)
    print('framecount:',framecount)

    #Save Video
    if not os.path.exists('./db'):
        os.makedirs('./db')
    start_time = datetime.now(timezone("Asia/Seoul")).strftime("_%y%m%d_%H%M%S")
    video_dir = f'./db/output{start_time}.MP4'
    out = cv2.VideoWriter(video_dir, fourcc, delay, (w,h))
    if not (out.isOpened()):
        print("File isn't opend!!")
        video.release()
        sys.exit()

    #Load frame and Save it
    start = time.time()
    timer = st.sidebar.empty()
    num_frames = 0
    while video.isOpened() and start_recording and not end_recording:
        ret, frame = video.read()

        sec = round(time.time() - start)
        timer.metric("Countdown", f"{sec//60:02d}:{sec%60:02d}")

        if ret and sec//60<number:
            num_frames += 1

            stframe.image(frame,channels='BGR', use_column_width=True)

            if start_recording:
                out.write(frame)
            
            cv2.waitKey(1)

        else:
            print("ret is false")
            break
    print('num frames:',num_frames)
    print()

    video.release()
    out.release()
    cv2.destroyAllWindows()
