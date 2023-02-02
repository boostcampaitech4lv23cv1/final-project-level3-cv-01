dataset_info = dict(
    dataset_name='aihub_sign_language',
    paper_info=dict(
        author='aihub',
        homepage='https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=103/',
    ),
    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='mid_shoulder',
            id=1,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        2:
        dict(
            name='right_shoulder',
            id=2,
            color=[0, 255, 0],
            type='upper',
            swap='left_shoulder'),
        3:
        dict(
            name='right_elbow',
            id=3,
            color=[0, 255, 0],
            type='upper',
            swap='left_elbow'),
        4:
        dict(
            name='right_wrist',
            id=4,
            color=[0, 255, 0],
            type='upper',
            swap='left_wrist'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='left_elbow',
            id=6,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        7:
        dict(
            name='left_wrist',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        8:
        dict(
            name='middle_hip',
            id=8,
            color=[255, 128, 0],
            type='lower',
            swap=''),
        9:
        dict(
            name='right_hip_1',
            id=9,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip_1'),
        10:
        dict(
            name='right_hip_2',
            id=10,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip_2'),
        11:
        dict(
            name='right_hip_3',
            id=11,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip_3'),
        12:
        dict(
            name='left_hip_1',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='right_hip_1'),
        13:
        dict(
            name='left_hip_2',
            id=13,
            color=[255, 128, 0],
            type='lower',
            swap='right_hip_2'),
        14:
        dict(
            name='left_hip_3',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='right_hip_3'),
        15:
        dict(
            name='right_eye',
            id=15,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        16:
        dict(
            name='left_eye',
            id=16,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        17:
        dict(
            name='right_cheek',
            id=17,
            color=[51, 153, 255],
            type='upper',
            swap='left_cheek'),
        18:
        dict(
            name='left_cheek',
            id=18,
            color=[51, 153, 255],
            type='upper',
            swap='right_cheek'),
        19:
        dict(
            name='left_hip_4',
            id=19,
            color=[255, 128, 0],
            type='lower',
            swap='right_hip_4'),
        20:
        dict(
            name='left_hip_5',
            id=20,
            color=[255, 128, 0],
            type='lower',
            swap='right_hip_5'),
        21:
        dict(
            name='left_hip_6',
            id=21,
            color=[255, 128, 0],
            type='lower',
            swap='right_hip_6'),
        22:
        dict(
            name='right_hip_4',
            id=22,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip_4'),
        23:
        dict(
            name='left_hip_5',
            id=23,
            color=[255, 128, 0],
            type='lower',
            swap='right_hip_5'),
        24:
        dict(
            name='left_hip_6',
            id=24,
            color=[255, 128, 0],
            type='lower',
            swap='right_hip_6'),
    },
    skeleton_info={
        0:
        dict(link=('left_cheek', 'left_eye'), id=0, color=[51, 153, 255]),
        1:
        dict(link=('left_eye', 'nose'), id=1, color=[51, 153, 255]),
        2:
        dict(link=('nose', 'right_eye'), id=2, color=[51, 153, 255]),
        3:
        dict(link=('right_eye', 'right_cheek'), id=3, color=[51, 153, 255]),
        4:
        dict(link=('left_wrist', 'left_elbow'), id=4, color=[0, 255, 0]),
        5:
        dict(link=('left_elbow', 'left_shoulder'), id=5, color=[0, 255, 0]),
        6:
        dict(link=('left_shoulder', 'mid_shoulder'), id=6, color=[0, 255, 0]),
        7:
        dict(link=('mid_shoulder', 'right_shoulder'), id=7, color=[0, 255, 0]),
        8:
        dict(link=('right_shoulder', 'right_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(link=('right_elbow', 'right_wrist'), id=9, color=[0, 255, 0]),
        10:
        dict(link=('left_hip_1', 'left_hip_2'), id=10, color=[255, 128, 0]),
        11:
        dict(link=('left_hip_2', 'left_hip_3'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_hip_3', 'left_hip_4'), id=12, color=[255, 128, 0]),
        13:
        dict(link=('left_hip_4', 'left_hip_5'), id=13, color=[255, 128, 0]),
        14:
        dict(link=('left_hip_4', 'left_hip_6'), id=14, color=[255, 128, 0]),
        15:
        dict(link=('middle_hip', 'left_hip_1'), id=15, color=[255, 128, 0]),
        16:
        dict(link=('middle_hip', 'right_hip_1'), id=16, color=[255, 128, 0]),
        17:
        dict(link=('right_hip_1', 'right_hip_2'), id=17, color=[255, 128, 0]),
        18:
        dict(link=('right_hip_2', 'right_hip_3'), id=18, color=[255, 128, 0]),
        19:
        dict(link=('right_hip_3', 'right_hip_4'), id=19, color=[255, 128, 0]),
        20:
        dict(link=('right_hip_4', 'right_hip_5'), id=20, color=[255, 128, 0]),
        21:
        dict(link=('right_hip_4', 'right_hip_6'), id=21, color=[255, 128, 0]),
    },
    joint_weights=[1.] * 25,
    sigmas=[],
    )
