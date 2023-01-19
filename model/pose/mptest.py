import mediapipe as mp  # Import mediapipe
import cv2  # Import opencv
import os
import json

import numpy as np
import pandas as pd
from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles


def calculate_angle(a, b):
    a = np.array(a)
    b = np.array(b)

    radians = np.arctan2(b[1] - a[1], b[0] - a[0])
    angle = np.abs(radians * 180.0 / np.pi)

    return angle

VIDEO_PATH = os.path.join('./db','output_230119_194953.webm')
def run(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # cap.get(cv2.CAP_PROP_FPS) == 30
    fps = 60
    cap.set(cv2.CAP_PROP_FPS,fps)
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    print(fps) 

    out = cv2.VideoWriter("./db/pose.webm", fourcc, fps, (width, height))

    anomaly = {"shoulder": [], "hand": []}
    shoulder_components={"start":[],"end":[],"elapsed":[]}
    hand_components={"time":[]}

    # Initiate holistic model
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        start_time = datetime.now()
        print(start_time)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make Detections
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Right hand
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(80, 22, 10), thickness=2, circle_radius=4
                ),
                mp_drawing.DrawingSpec(
                    color=(80, 44, 121), thickness=2, circle_radius=2
                ),
            )

            # Left Hand
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(121, 22, 76), thickness=2, circle_radius=4
                ),
                mp_drawing.DrawingSpec(
                    color=(121, 44, 250), thickness=2, circle_radius=2
                ),
            )

            # Pose Detections
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(245, 117, 66), thickness=2, circle_radius=4
                ),
                mp_drawing.DrawingSpec(
                    color=(245, 66, 230), thickness=2, circle_radius=2
                ),
            )
            out.write(image)

            # Export coordinates
            try:
                current_time = datetime.now()
                target_time = current_time - start_time
                landmarks = results.pose_landmarks.landmark

                # Get shoulder coordinates
                left_shoulder = [
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                ]
                right_shoulder = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                ]
                left_wrist = [
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y,
                ]
                right_wrist = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                ]


                # Calculate shoulder angle
                angle = calculate_angle(left_shoulder, right_shoulder)
                if angle <= 170 :  # 기준 정해야함
                    anomaly["shoulder"].append((target_time, datetime.now()))
                else:
                    shoulder_seconds = (
                        anomaly["shoulder"][-1][1] - anomaly["shoulder"][0][1]
                    ).total_seconds()
                    shoulder_anomaly_start = anomaly["shoulder"][0][0].total_seconds()
                    shoulder_anomaly_end = anomaly["shoulder"][-1][0].total_seconds()
                    if shoulder_seconds >= 1e-3:
                        shoulder_components["start"].append(shoulder_anomaly_start)
                        shoulder_components["end"].append(shoulder_anomaly_end)
                        shoulder_components["elapsed"].append(shoulder_seconds)
                    anomaly["shoulder"] = []

                # if results.left_hand_landmarks or results.right_hand_landmarks:
                #     hand_components["time"].append(target_time.total_seconds())
                if left_wrist or right_wrist:
                    hand_components["time"].append(target_time.total_seconds())

            except:
                pass


            cv2.imshow("Video Feed", image)
            

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    return shoulder_components, hand_components


def dict_to_json(d:dict):
    d_df = pd.DataFrame(d)
    d_json = d_df.to_json(orient='records')
    return d_json
    


if __name__ == "__main__":
    shoulder_info, hand_info = run(VIDEO_PATH) 
    shoulder_json,hand_json = dict_to_json(shoulder_info), dict_to_json(hand_info)
    # shoulder_response,hand_response = JSONResponse(json.loads(shoulder_json)),JSONResponse(json.loads(hand_json))
    print(shoulder_json,hand_json)