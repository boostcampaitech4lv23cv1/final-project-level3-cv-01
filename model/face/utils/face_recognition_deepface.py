from deepface import DeepFace
import cv2
import glob
import os
import mmcv
import pandas as pd
import argparse


def parse_args():
    """
    argparse 이용해서 실행 할 수 있게 추후 추가 예정
    """
    parser = argparse.ArgumentParser(description="Analyze emotion and Remake Video")
    parser.add_argument("--video_path", help="video path to analyze")
    parser.add_argument("--saved_dir", help="dir to save frame")
    parser.add_argument("--new_video_name", help="video name to save")
    args = parser.parse_args()
    return args


def video_to_frame(VIDEO_PATH, SAVED_DIR):

    if not os.path.exists(SAVED_DIR):
        os.makedirs(SAVED_DIR)

    cap = cv2.VideoCapture(VIDEO_PATH)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # fps = cap.get(cv2.CAP_PROP_FPS) / 40
    # print(cap.get(cv2.CAP_PROP_FPS))
    # print(fps)
    fps = 1

    while True:  # 무한 루프
        ret, frame = cap.read()  # 두 개의 값을 반환하므로 두 변수 지정

        if not ret:  # 새로운 프레임을 못받아 왔을 때 braek
            break
        if int(cap.get(1)) % int(fps) == 0:
            cv2.imwrite(SAVED_DIR + "/frame%06d.jpg" % count, frame)
            # print("Saved frame number : ", str(int(cap.get(1))))
            count += 1

    cap.release()  # 사용한 자원 해제

    frames = glob.glob(f"{SAVED_DIR}/*.jpg")
    frames.sort()

    return frames


def analyze_emotion(frames):
    models = {}
    models["emotion"] = DeepFace.build_model("Emotion")
    models["gender"] = DeepFace.build_model("Gender")
    emotions_mtcnn = DeepFace.analyze(
        img_path=frames,
        actions=("gender", "emotion"),
        models=models,
        enforce_detection=False,
        detector_backend="mtcnn",
    )

    return emotions_mtcnn


def make_emotion_df(emotions_mtcnn):
    angry = []
    disgust = []
    fear = []
    happy = []
    sad = []
    surprise = []
    neutral = []
    lenoflist = len(emotions_mtcnn)
    dominant_emotion = []

    for i in range(1, lenoflist + 1):
        tmp = "instance_" + str(i)
        angry.append(emotions_mtcnn[tmp]["emotion"]["angry"])
        disgust.append(emotions_mtcnn[tmp]["emotion"]["disgust"])
        fear.append(emotions_mtcnn[tmp]["emotion"]["fear"])
        happy.append(emotions_mtcnn[tmp]["emotion"]["happy"])
        sad.append(emotions_mtcnn[tmp]["emotion"]["sad"])
        surprise.append(emotions_mtcnn[tmp]["emotion"]["surprise"])
        neutral.append(emotions_mtcnn[tmp]["emotion"]["neutral"])
    df_mtcnn = pd.DataFrame(
        {
            "angry": angry,
            "disgust": disgust,
            "fear": fear,
            "happy": happy,
            "sad": sad,
            "surprise": surprise,
            "neutral": neutral,
        }
    )

    return df_mtcnn


def make_binary_df(emotions_mtcnn, df_mtcnn):
    pos_emo = ["happy", "neutral"]
    neg_emp = ["angry", "disgust", "fear", "sad", "surprise"]
    highest = []
    for i in range(len(df_mtcnn)):
        string = df_mtcnn.iloc[i].idxmax()
        highest.append(string)
    positive = []
    negative = []

    for i in range(1, len(emotions_mtcnn) + 1):
        tmp = "instance_" + str(i)
        p = 0
        n = 0
        if highest[i - 1] in pos_emo:
            p += emotions_mtcnn[tmp]["emotion"]["happy"]
            p += emotions_mtcnn[tmp]["emotion"]["neutral"]

        else:
            n += emotions_mtcnn[tmp]["emotion"]["angry"]
            n += emotions_mtcnn[tmp]["emotion"]["disgust"]
            n += emotions_mtcnn[tmp]["emotion"]["fear"]
            n += emotions_mtcnn[tmp]["emotion"]["sad"]
            n += emotions_mtcnn[tmp]["emotion"]["surprise"]
        positive.append(p)
        negative.append(n)
    df_binary = pd.DataFrame({"positive": positive, "negative": negative})
    return df_binary


def add_emotion_on_frame(emotions_mtcnn, df_mtcnn, saved_dir):
    len_of_df = len(df_mtcnn)
    text_of_rec = []
    for i in range(len_of_df):
        string = (
            df_mtcnn.iloc[i].idxmax()
            + "_"
            + str(round(df_mtcnn.iloc[i][df_mtcnn.iloc[i].idxmax()], 3))
            + "%"
        )
        text_of_rec.append(string)

    regions = []
    for i in range(1, len_of_df + 1):
        tmp = "instance_" + str(i)
        region = emotions_mtcnn[tmp]["region"]
        regions.append(region)

    images = glob.glob(f"{saved_dir}/*.jpg")
    images.sort()

    rec_image_list = []
    for idx, (region, i) in enumerate(zip(regions, images)):
        pth = cv2.imread(i)
        rec = (region["x"], region["y"], region["w"], region["h"])
        x = rec[0]
        y = rec[1] - 10
        pos = (x, y)
        rec_image = cv2.rectangle(pth, rec, (0, 255, 0), thickness=4)
        rec_image = cv2.putText(
            rec_image,
            text_of_rec[idx],
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (36, 255, 12),
            3,
        )
        rec_image_list.append(rec_image)
    return rec_image_list


def add_emotion_on_frame_new(df):

    len_of_df = len(df)
    rec_image_list = []

    for i in range(len_of_df):
        info = df.loc[i, :]
        string = info['emotion']
        pth = cv2.imread(info['frame'])
        rec = (info['x'], info['y'], info['w'], info['h'])
        x = rec[0]
        y = rec[1]
        pos = (x, y-10)
        rec_image = cv2.rectangle(pth, rec, (0, 255, 0), thickness=4)
        rec_image = cv2.putText(
            rec_image,
            string,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (36, 255, 12),
            3,
        )

        rec_image_list.append(rec_image)

    return rec_image_list


def frame_to_video(rec_image_list, video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"vp80")
    
    vid_save_name = f"./{video_path.split('/')[1]}/{video_path.split('/')[2]}/face_{video_path.split('/')[-1]}"
    out = cv2.VideoWriter(vid_save_name, fourcc, fps, (width, height))

    for rec_frame in rec_image_list:
        out.write(rec_frame)

    cap.release()
    out.release()

    return vid_save_name
