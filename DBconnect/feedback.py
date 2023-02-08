from pymongo import MongoClient
import json
import pandas as pd
'''
get feedback of result from streamlit
'''
class FeedbackDB:
    
    def __init__(self, video_dir, data): 
        ''' 
            video_dir: path to folder 
            data: dictionary type
        ''' 
        
        mongodb_URI = "mongodb+srv://heyi:20230214@hey-i.o4iunhl.mongodb.net/test"
        client = MongoClient(mongodb_URI)
        self.db = client['heyi']
        self.data = data
        self.path = video_dir

    def save_data(self,):
        
        data = self.analyze_feedback()
        print(f'업로드 할 데이터 정보:{data}')
        self.db.feedback.insert_one(data)
        print('피드백 업로드 완료.')
        
    def analyze_feedback(self, ):
        
        result = {'info':{'video_dir': self.path},
                  'data':{}}
        data_list = []
        
        face_data = self.db.face.find_one({'info.video_dir': self.path})
        frame_lst = face_data['data']
        
        face, pose, eye = self.check_timeline()
        
        for i, frame in enumerate(frame_lst):
            dic = {}
            dic['frame'] = frame['frame']
            
            if face[i] is not None:
                dic['face'] = face[i]
            if pose[i] is not None:
                dic['pose'] = pose[i]
            if eye[i] is not None:
                dic['eye'] = eye[i]
            
            data_list.append(dic)
        
        result['data'] = data_list
        print('피드백 분석 완료.')
        return result
                
    def check_timeline(self,):
        
        face_lst= [None for i in range(999999)]
        pose_lst= [None for i in range(999999)]
        eye_lst= [None for i in range(999999)]
        
        key_lst = self.data.keys()
        val_lst = self.data.values()
        print("타임라인 추출 시작")
        for k, v in zip(key_lst, val_lst):
            
            # filtering for xxx_all
            if len(key_lst.split('_')) > 3:
                if k.split('_')[0] == 'face':
                    start = int(k.split('_')[2])
                    end = int(k.split('_')[3])
                    for i in range(start,end+1):
                        face_lst[i] = v
                        
                elif k.split('_')[0] == 'pose':
                    start = int(k.split('_')[2])
                    end = int(k.split('_')[3])
                    for i in range(start,end+1):
                        pose_lst[i] = v
                        
                elif k.split('_')[0] == 'eye':
                    start = int(k.split('_')[2])
                    end = int(k.split('_')[3])
                    for i in range(start,end+1):
                        eye_lst[i] = v
                else:
                    print('입력 데이터 형식이 이상합니다..!')
        
        print(face_lst, pose_lst, eye_lst)
        print("타임라인 추출 완료.")           
        return face_lst, pose_lst, eye_lst
        