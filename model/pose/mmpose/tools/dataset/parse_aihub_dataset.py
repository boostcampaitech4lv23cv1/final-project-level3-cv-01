# Copyright (c) OpenMMLab. All rights reserved.
import csv
import json
import os
import time

import cv2
import numpy as np

np.random.seed(0)


def get_poly_area(x, y):
    """Calculate area of polygon given (x,y) coordinates (Shoelace formula)

    :param x: np.ndarray(N, )
    :param y: np.ndarray(N, )
    :return: area
    """
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def get_seg_area(segmentations):
    area = 0
    for segmentation in segmentations:
        area += get_poly_area(segmentation[:, 0], segmentation[:, 1])
    return area


def save_coco_anno(
    data_annotation, img_root, save_path, start_img_id=0, start_ann_id=0, kpt_num=25
):
    """Save annotations in coco-format.

    :param data_annotation: list of data annotation.
    :param img_root: the root dir to load images.
    :param save_path: the path to save transformed annotation file.
    :param start_img_id: the starting point to count the image id.
    :param start_ann_id: the starting point to count the annotation id.
    :param kpt_num: the number of keypoint.
    """
    images = []
    annotations = []

    img_id = start_img_id
    ann_id = start_ann_id

    for i in range(0, len(data_annotation)):
        data_anno = data_annotation[i]
        image_name = data_anno[0]

        img = cv2.imread(os.path.join(img_root, image_name))

        kp_string = data_anno[1]
        kps = json.loads(kp_string)

        seg_string = data_anno[2]
        segs = json.loads(seg_string)

        for kp, seg in zip(kps, segs):
            keypoints = np.zeros([kpt_num, 3])
            for ind, p in enumerate(kp):
                if p["position"] is None:
                    continue
                else:
                    keypoints[ind, 0] = p["position"][0]
                    keypoints[ind, 1] = p["position"][1]
                    keypoints[ind, 2] = 2

            segmentations = []

            max_x = -1
            max_y = -1
            min_x = 999999
            min_y = 999999
            for segm in seg:
                if len(segm["segment"]) == 0:
                    continue

                segmentation = np.array(segm["segment"])
                segmentations.append(segmentation)

                _max_x, _max_y = segmentation.max(0)
                _min_x, _min_y = segmentation.min(0)

                max_x = max(max_x, _max_x)
                max_y = max(max_y, _max_y)
                min_x = min(min_x, _min_x)
                min_y = min(min_y, _min_y)

            anno = {}
            anno["keypoints"] = keypoints.reshape(-1).tolist()
            anno["image_id"] = img_id
            anno["id"] = ann_id
            anno["num_keypoints"] = int(sum(keypoints[:, 2] > 0))
            anno["bbox"] = [
                float(min_x),
                float(min_y),
                float(max_x - min_x + 1),
                float(max_y - min_y + 1),
            ]
            anno["iscrowd"] = 0
            anno["area"] = get_seg_area(segmentations)
            anno["category_id"] = 1
            anno["segmentation"] = [seg.reshape(-1).tolist() for seg in segmentations]

            annotations.append(anno)
            ann_id += 1

        image = {}
        image["id"] = img_id
        image["file_name"] = image_name
        image["height"] = img.shape[0]
        image["width"] = img.shape[1]

        images.append(image)
        img_id += 1

    cocotype = {}

    cocotype["info"] = {}
    cocotype["info"]["description"] = "Aihub Dataset"
    cocotype["info"]["version"] = "1.0"
    cocotype["info"]["year"] = time.strftime("%Y", time.localtime())
    cocotype["info"]["date_created"] = time.strftime("%Y/%m/%d", time.localtime())

    cocotype["images"] = images
    cocotype["annotations"] = annotations
    cocotype["categories"] = [
        {
            "supercategory": "person",
            "id": 1,
            "name": "aihub",
            "keypoints": [
                "nose",
                "mid_shoulder",
                "right_shoulder",
                "right_elbow",
                "right_wrist",
                "left_shoulder",
                "left_elbow",
                "left_wrist",
                "middle_hip",
                "right_hip_1",
                "right_hip_2",
                "right_hip_3",
                "left_hip_1",
                "left_hip_2",
                "left_hip_3",
                "right_eye",
                "left_eye",
                "right_cheek",
                "left_cheek",
                "left_hip_4",
                "left_hip_5",
                "left_hip_6",
                "right_hip_4",
                "right_hip_5",
                "right_hip_6",
            ],
            "skeleton": [
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
            ],
        }
    ]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    json.dump(cocotype, open(save_path, "w"), indent=4)
    print("number of images:", img_id)
    print("number of annotations:", ann_id)
    print(f"done {save_path}")


dataset_dir = "/data/aihub/"
with open(os.path.join(dataset_dir, "annotations.csv"), "r") as fp:
    data_annotation_all = list(csv.reader(fp, delimiter=","))[1:]

np.random.shuffle(data_annotation_all)

data_annotation_train = data_annotation_all[0:9440]
data_annotation_val = data_annotation_all[9440:]

img_root = os.path.join(dataset_dir, "images")
save_coco_anno(
    data_annotation_train,
    img_root,
    os.path.join(dataset_dir, "annotations", "aihub_train.json"),
    kpt_num=25,
)
save_coco_anno(
    data_annotation_val,
    img_root,
    os.path.join(dataset_dir, "annotations", "aihub_test.json"),
    start_img_id=12500,
    start_ann_id=15672,
    kpt_num=25,
)
