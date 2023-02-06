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

# annotations = annotations[1:]

indexes = range(20, 111)
total_annotations = []
for annotation in tqdm.tqdm(annotations):
    with open(annotation) as a:
        anno = json.load(a)
        # 1920 x 1080 -> 320 x 180
        x_kpts = anno["people"]["pose_keypoints_2d"][::3]
        y_kpts = anno["people"]["pose_keypoints_2d"][1::3]

        needed_x_kpts = np.array(x_kpts[:8] + x_kpts[15:19]) / 6
        needed_y_kpts = np.array(y_kpts[:8] + y_kpts[15:19]) / 6

        needed_xy = list(zip(needed_x_kpts, needed_y_kpts))
        total_annotations.append(needed_xy)
annotations_amount = len(total_annotations)
total_annotations = np.array(total_annotations)
print("total_annotations shape: ", total_annotations.shape)


total_annotated = [[True for _ in range(12)] for _ in range(annotations_amount)]
total_annotated = np.array(total_annotated)
print("total_annotated shape: ", total_annotated.shape)


total_images = []
for image in tqdm.tqdm(images):
    img = cv2.imread(image)
    H, W, C = img.shape

    img = cv2.resize(img, (W // 6, H // 6), interpolation=cv2.INTER_CUBIC)
    total_images.append(img)
total_images = np.array(total_images)
print("total_images shape: ", total_images.shape)

skeletons = [
    [11, 9],
    [9, 0],
    [0, 8],
    [8, 10],
    [7, 6],
    [6, 5],
    [5, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 1],
    [9, 8],
]
skeletons = np.array(skeletons)
print("skeletons shape: ", skeletons.shape)

with h5py.File("../aihub/aihub_annotations.h5", "w") as f:
    f.create_dataset(
        "annotations", total_annotations.shape, compression="gzip", compression_opts=9
    )
    f.create_dataset(
        "annotated", total_annotated.shape, compression="gzip", compression_opts=9
    )
    f.create_dataset(
        "images", total_images.shape, compression="gzip", compression_opts=9
    )
    f.create_dataset(
        "skeleton", skeletons.shape, compression="gzip", compression_opts=9
    )

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
