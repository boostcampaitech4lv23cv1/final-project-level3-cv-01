dataset_info = dict(
    dataset_name="aihub_sign_language",
    paper_info=dict(
        author="aihub",
        homepage="https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=103/",
    ),
    keypoint_info={
        0: dict(name="nose", id=0, color=[51, 153, 255], type="upper", swap=""),
        1: dict(name="mid_shoulder", id=1, color=[0, 255, 0], type="upper", swap=""),
        2: dict(
            name="right_shoulder",
            id=2,
            color=[0, 255, 0],
            type="upper",
            swap="left_shoulder",
        ),
        3: dict(
            name="right_elbow", id=3, color=[0, 255, 0], type="upper", swap="left_elbow"
        ),
        4: dict(
            name="right_wrist", id=4, color=[0, 255, 0], type="upper", swap="left_wrist"
        ),
        5: dict(
            name="left_shoulder",
            id=5,
            color=[0, 255, 0],
            type="upper",
            swap="right_shoulder",
        ),
        6: dict(
            name="left_elbow", id=6, color=[0, 255, 0], type="upper", swap="right_elbow"
        ),
        7: dict(
            name="left_wrist", id=7, color=[0, 255, 0], type="upper", swap="right_wrist"
        ),
        8: dict(
            name="right_eye", id=15, color=[51, 153, 255], type="upper", swap="left_eye"
        ),
        9: dict(
            name="left_eye", id=16, color=[51, 153, 255], type="upper", swap="right_eye"
        ),
        10: dict(
            name="right_cheek",
            id=17,
            color=[51, 153, 255],
            type="upper",
            swap="left_cheek",
        ),
        11: dict(
            name="left_cheek",
            id=18,
            color=[51, 153, 255],
            type="upper",
            swap="right_cheek",
        ),
    },
    skeleton_info={
        0: dict(link=("left_cheek", "left_eye"), id=0, color=[51, 153, 255]),
        1: dict(link=("left_eye", "nose"), id=1, color=[51, 153, 255]),
        2: dict(link=("nose", "right_eye"), id=2, color=[51, 153, 255]),
        3: dict(link=("right_eye", "right_cheek"), id=3, color=[51, 153, 255]),
        4: dict(link=("left_wrist", "left_elbow"), id=4, color=[0, 255, 0]),
        5: dict(link=("left_elbow", "left_shoulder"), id=5, color=[0, 255, 0]),
        6: dict(link=("left_shoulder", "mid_shoulder"), id=6, color=[0, 255, 0]),
        7: dict(link=("mid_shoulder", "right_shoulder"), id=7, color=[0, 255, 0]),
        8: dict(link=("right_shoulder", "right_elbow"), id=8, color=[0, 255, 0]),
        9: dict(link=("right_elbow", "right_wrist"), id=9, color=[0, 255, 0]),
        10: dict(link=("nose", "mid_shoulder"), id=10, color=[0, 255, 0]),
        11: dict(link=("left_eye", "right_eye"), id=11, color=[51, 153, 255]),
    },
    joint_weights=[1.0] * 12,
    sigmas=[
        0.026,
        0.079,
        0.079,
        0.072,
        0.062,
        0.079,
        0.072,
        0.062,
        0.025,
        0.025,
        0.035,
        0.035,
    ],
)
