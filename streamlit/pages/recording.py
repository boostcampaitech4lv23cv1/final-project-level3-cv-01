import sys
import cv2
import streamlit as st

# stop = st.button('Stop Recording Video')
# stop = st.radio(
#         "Recording Start or Stop",
#         ["Record Start", "Record Stop"],
#         horizontal=True,
#     )
st.markdown(
    '''
    # 면접 자세 분석을 위해 녹화할게요!
    ### 준비가 되면 녹화 버튼을 눌러주세요
    '''
)
start = st.button('Start Recording')
stframe = st.empty()

# if stop == "Record Start":
if start:
    st.markdown(
        '''
        #### 질문 : 자기 소개를 1분 안에 해주세요
        '''
    )
    stop = st.button('Stop Recording')
    while not stop:

        with st.spinner('get ready for camera'):

            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print("Camera open failed!")
                st.write("Camera open failed!")
                sys.exit()

            w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'X264') # *'DIVX' == 'D', 'I', 'V', 'X'
            delay = round(1000 / fps)

            out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

            # if not out.isOpened():
            #     print('File open failed!')
            #     cap.release()
            #     sys.exit()

        with st.spinner('Recording!'):

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # inversed = ~frame

                out.write(frame)

                # cv2.imshow('frame', frame)
                stframe.image(frame, channels = 'BGR')
                # cv2.imshow('inversed', inversed)

                # if cv2.waitKey(delay) == 27:
                #     break

# cap.release()
# out.release()
# cv2.destroyAllWindows()