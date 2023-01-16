import sys
import cv2
import numpy as np
import streamlit as st

st.set_page_config(layout='wide')
st.markdown('# 면접 자세를 확인하세요')
st.markdown('## 어깨와 얼굴을 선에 맞춰주세요')

col1, col2 = st.columns(2)

with col2:
    st.markdown(
        ''' 
        ### 올바른 면접 자세란?
        - 허리를 피세요
        - 어깨를 피세요
        - 웃음꽃 피세요
        '''
        )

with col1:

    stframe = st.empty()

    with st.spinner('Get ready for camera'):

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Camera open failed!")
            st.write("Camera open failed!")
            sys.exit()

        w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        person = np.array(cv2.resize(cv2.imread('streamlit/person.png'), (w, h))) / 255
        person = np.array(person, dtype = 'u1')


    # fourcc = cv2.VideoWriter_fourcc(*'X264') # *'DIVX' == 'D', 'I', 'V', 'X'
    # delay = round(1000 / fps)

    # out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

                # if not out.isOpened():
                #     print('File open failed!')
                #     cap.release()
                #     sys.exit()
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # inversed = ~frame

        # cv2.imshow('frame', frame)
        stframe.image(frame * person, channels = 'BGR')
        # cv2.imshow('inversed', inversed)

        # if cv2.waitKey(delay) == 27:
        #     break

# cap.release()
# out.release()
# cv2.destroyAllWindows()