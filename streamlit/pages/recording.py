import os
import cv2
import sys
import time
import tempfile
from pytz import timezone
from datetime import datetime
import streamlit as st

# Eye Tracking
from gaze_tracking import GazeTracking
gaze = GazeTracking()

# st.session_state.start_recording = False
# st.session_state.end_recording = False

# print(st.session_state)

# Basic App Scaffolding
st.title('HEY-I')
st.subheader('면접 영상을 녹화하세요')
st.markdown('##### 선택한 시간이 지나거나 End Recording 버튼을 누르면 녹화가 종료됩니다.')

# Create Sidebar
st.sidebar.title('Settings')

## Get Video
temp_file = tempfile.NamedTemporaryFile(delete=False)

number = st.sidebar.number_input('분 입력',1,10)
start_recording = st.sidebar.button('Start Recordinging')
# start_recording = st.sidebar.button('Start Recordinging', key='start_recording')

if start_recording:
    # print(st.session_state.start_recording)
    st.markdown('**질문** : 1분 자기 소개를 해주세요')
    stframe = st.empty()
    with st.spinner('Get Ready for Camera'):
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
        # delay = 6
        print('fps:',fps)
        print('framecount:',framecount)

        #Save Video
        if not os.path.exists('./db'):
            os.makedirs('./db')
        start_time = datetime.now(timezone("Asia/Seoul")).strftime("_%y%m%d_%H%M%S")
        video_dir = f'./db/output{start_time}.mp4'
        st.session_state.video_dir = video_dir
        out = cv2.VideoWriter(video_dir, fourcc, fps, (w,h))
        if not (out.isOpened()):
            print("File isn't opend!!")
            video.release()
            sys.exit()

    end_recording = st.sidebar.button('End Recordinging')
    # end_recording = st.sidebar.button('End Recordinging', key='end_recording')

    #Load frame and Save it
    start = time.time()
    timer = st.sidebar.empty()
    num_frames = 0
    while video.isOpened() and start_recording and not end_recording:
        ret, frame = video.read()

        ### eye tracking ###
        gaze.refresh(frame)
        
        frame = gaze.annotated_frame()
        text = ""

        if gaze.is_blinking():
            text = "Blinking"
        elif gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"

        cv2.putText(frame, text, (90, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 500), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)



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

# print(st.session_state)

# if end_recording and os.path.exists(st.session_state.video_dir):
#     st.write(f'{video_dir}에 면접 영상이 저장되었습니다. 수고하셨습니다!')

if 'video_dir' in st.session_state.keys():
    if os.path.exists(st.session_state.video_dir):
        # print(st.session_state.video_dir)
        video_file = open(st.session_state.video_dir, 'rb')
        video_bytes = video_file.read()
        st.write('가장 최근 녹화된 영상입니다. 이 영상으로 업로드 할 것인지 결정해주세요')
        check = st.button('Check Video')
        if check:
            st.video(video_bytes)
