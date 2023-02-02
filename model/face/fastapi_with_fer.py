import os
import cv2
import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
#from fer_pl import LightningModel
from model.face.fer_pl import LightningModel
from model.face.dataset_pl import testDataset
#from dataset_pl import testDataset

idx_to_class = {
    0: "angry",
    1: "anxiety",
    2: "happy",
    3: "hurt",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

def video_to_frame(VIDEO_PATH, SAVED_DIR):

    if not os.path.exists(SAVED_DIR):
        os.makedirs(SAVED_DIR)

    cap = cv2.VideoCapture(VIDEO_PATH)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:  # 무한 루프
        ret, frame = cap.read()  # 두 개의 값을 반환하므로 두 변수 지정

        if not ret:  # 새로운 프레임을 못받아 왔을 때 braek
            break
        if int(cap.get(1)) % int(fps) == 0:
            cv2.imwrite(SAVED_DIR + "/frame%d.jpg" % count, frame)
            print("Saved frame number : ", str(int(cap.get(1))))
            count += 1

        # 10ms 기다리고 다음 프레임으로 전환, Esc누르면 while 강제 종료
        if cv2.waitKey(10) == 27:
            break

    cap.release()  # 사용한 자원 해제
    cv2.destroyAllWindows()

    frames = glob.glob(f"{SAVED_DIR}/*.jpg")
    frames.sort()

    return frames


def analyze_emotion(frames_dir, model_ckpt_name="./models/best_val_posneg_acc.ckpt", batch_size=8):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = LightningModel.load_from_checkpoint(model_ckpt_name)
    model = model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    testdataset = testDataset(frames_dir, transform)
    testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)

    bbox_dict = {}

    model.eval()
    with torch.no_grad():
        for i in tqdm(testloader):
            input, box, path = i
            input = input.to(device)
            pred = model(input)
            pred = pred.argmax(dim=1).cpu().tolist()

            for j in range(len(box[0])):
                bbox_dict[path[j]] = [
                    pred[j],
                    idx_to_class[pred[j]],
                    int(box[0][j]),
                    int(box[1][j]),
                    int(box[2][j]),
                    int(box[3][j]),
                ]

    return bbox_dict


def make_emotion_df(emotions_mtcnn):
    df = pd.DataFrame({"frame":emotions_mtcnn.keys(),"values":[r[1] for r in emotions_mtcnn.values()]})
    return df

def add_emotion_on_frame(emotions_mtcnn, saved_dir):
    regions = [value[2:] for value in emotions_mtcnn.values()]

    images = glob.glob(f"{saved_dir}/*.jpg")
    images.sort()

    emotions = [emotion[1] for emotion in emotions_mtcnn.values()]

    rec_image_list = []
    for idx, (region, img, emotion) in enumerate(zip(regions, images, emotions)):
        img = cv2.imread(img)
        xmin, ymin, xmax, ymax = region
        rec_image = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=3)
        cv2.putText(rec_image, emotion, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)
        rec_image_list.append(rec_image.copy())
    return rec_image_list


def frame_to_video(rec_image_list, video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"vp80")
    
    vid_save_name = f"./{video_path.split('/')[1]}/{video_path.split('/')[2]}/face_{video_path.split('/')[-1]}"
    #vid_path, vid_name = os.path.join(*video_path.split("/")[:-1]), video_path.split("/")[-1]
    #vid_name = "face_"+vid_name
    #vid_path = "/"+vid_path
    #vid_save_name = os.path.join(vid_path,vid_name)

    print("vid_save_name:",vid_save_name)
    out = cv2.VideoWriter(vid_save_name, fourcc, 2, (width, height))
    for rec_frame in rec_image_list:
        print(rec_frame.shape)
        out.write(rec_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def make_binary_df(emotions_mtcnn):
    pos_emo = ("happy", "neutral")
    #neg_emp = ("angry", "disgust", "fear", "sad", "surprise")
    binary_emotion = ["positive" if v in pos_emo else "negative" for v in emotions_mtcnn.values()]
    return binary_emotion