import h5py
import numpy as np
import os
import fnmatch
import cv2
import json
import tqdm

PATH = os.path.abspath(os.path.join("..", "aihub"))
SOURCE_IMAGES = os.path.join(PATH, "images")
SOURCE_ANNOTATIONS = os.path.join(PATH, "annotations")
images = []
annotations = []
for root, dirnames, filenames in os.walk(SOURCE_IMAGES):
    for filename in fnmatch.filter(filenames, "*.*"):
        images.append(os.path.join(root, filename))

for root, dirnames, filenames in os.walk(SOURCE_ANNOTATIONS):
    for filename in fnmatch.filter(filenames, "*.*"):
        annotations.append(os.path.join(root, filename))

annotations = annotations[1:]

total_annotations = []
for annotation in tqdm.tqdm(annotations):
    with open(annotation) as a:
        anno = json.load(a)
        # 1920 x 1080 -> 320 x 180
        x_points = np.array(anno["people"]["pose_keypoints_2d"][::3]) / 6
        y_points = np.array(anno["people"]["pose_keypoints_2d"][1::3]) / 6
        new_xy = list(zip(x_points, y_points))
        total_annotations.append(new_xy)
total_annotations = np.array(total_annotations)
print("total_annotations shape: ", np.array(total_annotations).shape)


total_annotated = [[True for _ in range(25)] for _ in range(12440)]
total_annotated = np.array(total_annotated)
print("total_annotated shape: ", np.array(total_annotated).shape)


total_images = []
for image in tqdm.tqdm(images):
    img = cv2.imread(image)
    H, W, C = img.shape

    img = cv2.resize(img, (W // 6, H // 6), interpolation=cv2.INTER_CUBIC)
    total_images.append(img)
total_images = np.array(total_images)
print("total_images shape: ", total_images.shape)

skeletons = [
    [18, 16],
    [16, 0],
    [0, 15],
    [15, 17],
    [7, 6],
    [6, 5],
    [5, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [12, 13],
    [13, 14],
    [14, 19],
    [19, 20],
    [19, 21],
    [8, 12],
    [8, 9],
    [9, 10],
    [10, 11],
    [11, 22],
    [22, 23],
    [22, 24],
    [0, 8],
    [23, 24],
    [20, 21],
]
skeletons = np.array(skeletons)
print("skeletons shape: ", skeletons.shape)

with h5py.File("../aihub/aihub_annotations.h5", "w") as f:
    f.create_dataset("annotations", total_annotations.shape)
    f.create_dataset("annotated", total_annotated.shape)
    f.create_dataset("images", total_images.shape)
    f.create_dataset("skeleton", skeletons.shape)

    f["annotations"][:] = total_annotations
    f["annotated"][:] = total_annotated
    f["images"][:] = total_images
    f["skeleton"][:] = skeletons

with h5py.File("../aihub/aihub_annotations.h5", "r") as hf:
    r_anns = np.array(hf["annotations"])
    r_anned = np.array(hf["annotated"])
    r_images = np.array(hf["images"])
    r_skeletons = np.array(hf["skeleton"])
    print(r_anns.shape, r_anned.shape, r_images.shape, r_skeletons.shape)
