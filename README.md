<img src="images/main.png" width="100%" height="80%"/>
<div align="center">
<img src="https://img.shields.io/badge/Python-grey?style=flat&logo=python&logoColor=p"/>
<img src="https://img.shields.io/badge/PyTorch-grey?style=flat&logo=PyTorch&logoColor=red"/>
<img src="https://img.shields.io/badge/PyTorch Lightning-grey?style=flat&logo=PyTorch Lightning&logoColor=purple"/>  
<img src="https://img.shields.io/badge/FastAPI-grey?style=flat&logo=FastAPI&logoColor=green"/>
<img src="https://img.shields.io/badge/Streamlit-grey?style=flat&logo=Streamlit&logoColor=red"/>
<img src="https://img.shields.io/badge/MongoDB-grey?style=flat&logo=mongodb&logoColor="/>
<img src="https://img.shields.io/badge/GCP-grey?style=flat&logo=Googlecloud&logoColor="/>
<img src="https://img.shields.io/badge/Docker-grey?style=flat&logo=docker&logoColor=docker"/>  
<img src="https://img.shields.io/badge/GitHub Actions-grey?style=flat&logo=github-actions&logoColor=docker"/>
<img src="https://img.shields.io/badge/Git-grey?style=flat&logo=Git&logoColor="/>
<img src="https://img.shields.io/badge/Airflow-grey?style=flat&logo=apache-airflow&logoColor=red"/>  
<img src="https://img.shields.io/badge/slack-grey?style=flat&logo=slack&logoColor=slack"/>
<img src="https://img.shields.io/badge/Notion-grey?style=flat&logo=notion&logoColor=notion"/>
<img src="https://img.shields.io/badge/WebRTC-grey?style=flat&logo=WebRTC&logoColor=WebRTC"/>
</div>

# 👨‍🏫 HEY-I (HElp Your Interview)
## Project Summary
- 면접 진행 시 행동 분석을 통한 면접 도우미
- Facial Expression Recognition, Pose Estimation, Eye Tracking 사용해 얼굴 표정, 자세, 시선 처리에 대한 변화 및 이상치 전달
***
## Contributors🔥
| [김범준](https://github.com/quasar529) | [백우열](https://github.com/wooyeolBaek) | [조용재](https://github.com/yyongjae) | [조윤재](https://github.com/KidsareBornStars) | [최명헌](https://github.com/MyeongheonChoi) |
| :-: | :-: | :-: | :-: | :-: |
| <img src="https://avatars.githubusercontent.com/quasar529" width="100"> | <img src="https://avatars.githubusercontent.com/wooyeolBaek" width="100"> | <img src="https://avatars.githubusercontent.com/yyongjae" width="100"> | <img src="https://avatars.githubusercontent.com/KidsareBornStars" width="100"> | <img src="https://avatars.githubusercontent.com/MyeongheonChoi" width="100"> 
| **Face - deepface, GCP - Cloud Storage & Cloud Run, Airflo** | **Pose - mediapipe, Backend - FastAPI, CI/CD - Github Action** | **Eye - gaze tracking, MongoDB, Backend - FastAPI** | **Pose - mediapipe&mmpose, CI/CD - Github Action** | **Face - Facial Emotion Recognition Modeling, Frontend- streamlit, Backend-FastAPI** |
***
## Architecture Flow Map

<img src="images/pipeline.png" width="100%" height="80%"/>

***
## Data Pipeline

<img src="images/data_pipeline.png" width="100%" height="80%"/>

***
## Demo

**녹화 준비**

<img src="images/prepare.gif" width="50%" height="50%"/>

**녹화 혹은 파일 업로드**

<img src="images/recording1.gif" width="50%" height="50%"/><img src="images/recording2.gif" width="50%" height="50%"/>

**전체 분석 결과**

<img src="images/result1.gif" width="50%" height="50%"/><img src="images/result2.gif" width="50%" height="50%"/><img src="images/result3.gif" width="50%" height="50%"/>

**세부 분석 결과**

<img src="images/user_feedback2.gif" width="50%" height="50%"/><img src="images/user_feedback3.gif" width="50%" height="50%"/><img src="images/user_feedback4.gif" width="50%" height="50%"/>

**피드백 전달**

<img src="images/feedback5.gif" width="50%" height="50%"/>

***
## Model

**Facial Emotion Recognition**
- Model : EfficientNet B0
- Dataset : <a href='https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=82'>한국인 감정인식을 위한 복합 영상</a>
- Metric : 
  - Accuracy - 7 classes : 0.6285
  - Accuracy - 2 classes : 0.9112


**Pose Estimation**
- Model : Resnet50
- Dataset : <a href='https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=103'>수어 영상</a>
- Metric : Ap
***
## Folder Structure 📂
```
├── 📄README.md
├── 📄requirements.txt
├── 📂airflow
│    └── 📂dags
├── 📂DBconnect
├── 📂model
│    ├── 📂eye
|    │    └── 📂gaze_tracking
│    ├── 📂face
│    │    ├── 📂models
│    │    └── 📂utils
│    └── 📂pose
│         └── 📂mmmpose
├── 📂FastAPI
└── 📂streamlit
     └── 📂pages
```
***
## Reference

- <a href='https://github.com/serengil/deepface'>Deepface</a>
- <a href='https://github.com/open-mmlab/mmpose'>mmpose</a>
- <a href='https://github.com/HSE-asavchenko/face-emotion-recognition'>emotion pretrained</a>
- <a href='https://github.com/antoinelame/GazeTracking'>Gaze Tracking</a>
- <a href='https://github.com/google/mediapipe'>mediapipe</a>
