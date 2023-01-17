import mediapipe as mp  # Import mediapipe
import cv2  # Import opencv

import numpy as np
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


def run():
    cap = cv2.VideoCapture("./db/output_230117_145243.mp4")

    anomaly = {"shoulder": [], "hand": []}
    shoulder_coordinates = {"left": [], "right": []}

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
                shoulder_coordinates["left"].append(
                    (f"{target_time.total_seconds():.3f}초", left_shoulder)
                )
                right_shoulder = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                ]
                shoulder_coordinates["right"].append(
                    (f"{target_time.total_seconds():.3f}초", right_shoulder)
                )

                # Calculate shoulder angle
                angle = calculate_angle(left_shoulder, right_shoulder)
                if angle <= 170 or angle >= 190:  # 기준 정해야함
                    anomaly["shoulder"].append((target_time, datetime.now()))
                else:
                    shoulder_seconds = (
                        anomaly["shoulder"][-1][1] - anomaly["shoulder"][0][1]
                    ).total_seconds()
                    shoulder_anomaly_start = anomaly["shoulder"][0][0].total_seconds()
                    shoulder_anomaly_end = anomaly["shoulder"][-1][0].total_seconds()
                    if shoulder_seconds >= 1e-3:
                        print(
                            f"{shoulder_anomaly_start:.3f}초 부터 {shoulder_anomaly_end:.3f}초 까지 {shoulder_seconds:.3f}초 동안 자세가 좋지 않았습니다."
                        )
                    anomaly["shoulder"] = []

                if results.left_hand_landmarks or results.right_hand_landmarks:
                    print(f"{target_time.total_seconds():.3f}초에 손이 나왔습니다.")

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


if __name__ == "__main__":
    run()
