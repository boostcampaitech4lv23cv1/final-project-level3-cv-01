# Copyright (c) OpenMMLab. All rights reserved.
import os
import pandas as pd
import warnings
from datetime import datetime
from argparse import ArgumentParser

from collections import defaultdict

import cv2
import mmcv
import numpy as np
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo


def calculate_angle(a, b):
    a = np.array(a)
    b = np.array(b)

    radians = np.arctan2(b[1] - a[1], b[0] - a[0])
    angle = np.abs(radians * 180.0 / np.pi)

    return angle


def main(video_path="./model/pose/recording.webm", out_video_root="./db"):
    pose_config = "model/pose/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py"
    pose_checkpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
    show = False
    device = "cuda:0"
    kpt_thr = 0.3
    radius = 4
    thickness = 1

    assert show or (out_video_root != "")
    print("Initializing model...")
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device.lower())

    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
        warnings.warn(
            "Please set `dataset_info` in the config."
            "Check https://github.com/open-mmlab/mmpose/pull/663 for details.",
            DeprecationWarning,
        )
    else:
        dataset_info = DatasetInfo(dataset_info)

    # read video
    video = mmcv.VideoReader(video_path)
    assert video.opened, f"Faild to load video file {video_path}"

    fps = video.fps
    size = (video.width, video.height)

    if out_video_root == "":
        save_out_video = False
    else:
        os.makedirs(out_video_root, exist_ok=True)
        save_out_video = True

    print("!!!!!save_out_video", out_video_root)
    save_dir = "/".join(out_video_root.split("/")[:-1]) +"/pose_"+os.path.basename(video_path)
    print("!!!!!videowriter path",save_dir)

    if save_out_video:
        #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"vp80")
        print("out_video_root",out_video_root)
        videoWriter = cv2.VideoWriter(
            save_dir,
            fourcc,
            fps,
            size,
        )

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    print("Running inference...")
    start_time = datetime.now()

    frame_cnt = 0
    result = defaultdict(list)
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        # keep the person class bounding boxes.
        person_results = [{"bbox": np.array([0, 0, size[0], size[1]])}]

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            cur_frame,
            person_results,
            format="xyxy",
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names,
        )
        keypoints = pose_results[0]["keypoints"]
        current_time = datetime.now()

        # frame 별로 번호주고, 해당 번호에서의 shoulder_angle, hands_on 값 저장
        # -> 나중에 fps frame 번호에 해당하는 동영상 초 계산해서 추가할 예정
        frame_cnt += 1
        result["frame_id"].append(frame_cnt)
        result['nose'].append(tuple(keypoints[0][:2]) if keypoints[0][:2].any() else (-1, -1))
        result['left_eye'].append(tuple(keypoints[1][:2]) if keypoints[1][:2].any() else (-1, -1))
        result['right_eye'].append((keypoints[2][:2]) if keypoints[2][:2].any() else (-1, -1))
        result['left_ear'].append((keypoints[3][:2]) if keypoints[3][:2].any() else (-1, -1))
        result['right_ear'].append((keypoints[4][:2]) if keypoints[4][:2].any() else (-1, -1))
        result['left_shoulder'].append((keypoints[5][:2]) if keypoints[5][:2].any() else (-1, -1))
        result['right_shoulder'].append((keypoints[6][:2]) if keypoints[6][:2].any() else (-1, -1))
        result['left_elbow'].append((keypoints[7][:2]) if keypoints[7][:2].any() else (-1, -1))
        result['right_elbow'].append((keypoints[8][:2]) if keypoints[8][:2].any() else (-1, -1))
        result['left_wrist'].append((keypoints[9][:2]) if keypoints[9][:2].any() else (-1, -1))
        result['right_wrist'].append((keypoints[10][:2]) if keypoints[10][:2].any() else (-1, -1))
        # shoulder_angle = round(calculate_angle(left_shoulder, right_shoulder))
        # result["shoulder_angle"].append(shoulder_angle)

        # left_wrist = keypoints[9][:2]
        # right_wrist = keypoints[10][:2]

        # hands_on = True if left_wrist.any() or right_wrist.any() else False
        # result["hands_on"].append(hands_on)

        # show the results
        vis_frame = vis_pose_result(
            pose_model,
            cur_frame,
            pose_results,
            radius=radius,
            thickness=thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=kpt_thr,
            show=False,
        )

        if show:
            cv2.imshow("Frame", vis_frame)

        if save_out_video:
            videoWriter.write(vis_frame)

        if show and cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if save_out_video:
        videoWriter.release()
    if show:
        cv2.destroyAllWindows()

    #df = pd.DataFrame({k: v for k, v in result.items()})

    #return df
    return dict(result)


if __name__ == "__main__":
    print(main())
    # main()
