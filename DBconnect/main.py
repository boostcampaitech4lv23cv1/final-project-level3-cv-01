from pymongo import MongoClient
import json
import pandas as pd
import numpy as np
import certifi


class UserDB:
    def __init__(self, name, num, date, video_dir): # name: name_phone, data: recording date, video_dir: GCS directory
        
        ca = certifi.where()
        mongodb_URI = "mongodb+srv://heyi:20230214@hey-i.o4iunhl.mongodb.net/test"
        self.client = MongoClient(mongodb_URI, tlsCAFile=ca)
        
        self.db = self.client['heyi']
        self.info = {}
        self.info['name'] = name
        self.info['num'] = num
        self.info['date'] = date
        self.info['video_dir'] = video_dir
        
    def save_data(self): 
        # save user information 
        self.db.user.insert_one(self.info)
    
    def load_data_inf(self):
        pass
    
    def load_data_train(self):
        pass
class FaceDB(UserDB):
    def __init__(self, path):
        
        mongodb_URI = "mongodb+srv://heyi:20230214@hey-i.o4iunhl.mongodb.net/test"
        self.client = MongoClient(mongodb_URI)
        self.db = self.client['heyi']
        self.path = path
    
    def save_data(self, data):
        # user info에 저장된 사용자의 정보를 가져온다.
        
        data_json = json.loads(data)
            
        dic = {
            'info': {
                'name': self.info['name'] ,
                'video_dir': self.info['video_dir']
            },
            
        }
        dic['data'] = data_json
        
        self.db.face.insert_one(dic)
    
    def load_data_inf(self):
        data = self.db.face.find_one({'info.video_dir': self.info['video_dir']})
        df_face = pd.DataFrame.from_dict(data['data'])
        
        return df_face
    
    def load_data_train(self):
        data = self.db.face.find_one({'info.video_dir': self.path})  
        try:
            df_face = pd.DataFrame.from_dict(data['data'])
            df_train = df_face.loc[:,['frame', 'emotion']]
            return df_train 
        except:
            return pd.DataFrame()
# class FaceDB(UserDB):
#     def __init__(self):
#         super().__init__()
#     def save_data(self, data):
#         # user info에 저장된 사용자의 정보를 가져온다.
        
#         data_json = json.loads(data)
            
#         dic = {
#             'info': {
#                 'name': self.info['name'] ,
#                 'video_dir': self.info['video_dir']
#             },
            
#         }
#         dic['data'] = data_json
        
#         self.db.face.insert_one(dic)
    
#     def load_data_inf(self):
#         data = self.db.face.find_one({'info.video_dir': self.info['video_dir']})
#         df_face = pd.DataFrame.from_dict(data['data'])
        
#         return df_face
    
#     def load_data_train(self, path):
#         data = self.db.face.find_one({'info.video_dir': path})
#         df_face = pd.DataFrame.from_dict(data['data'])
#         df_train = df_face.loc[:,['frame', 'emotion']]
    
#         return df_train
              
class PoseDB(UserDB):
      
    def save_data(self, data):
        # user info에 저장된 사용자의 정보를 가져온다.
        
        
        data_json = json.loads(data)
        
        dic = {
            'info': {
                'name': self.info['name'] ,
                'video_dir': self.info['video_dir']
            },
            'data': {}
        }
        dic['data'] = data_json
        
        self.db.pose.insert_one(dic)
        

    def load_data_inf(self, ):
        
        data = self.db.pose.find_one({'info.video_dir': self.info['video_dir']})
        df_pose = pd.DataFrame.from_dict(data['data'])
        
        return df_pose
    
    def load_data_train(self):
        pass
        
    
class EyeDB(UserDB):
    
    def save_data(self, data):
        
        data_json = json.loads(data)
            
        # user info에 저장된 사용자의 정보를 가져온다.
        dic = {
            'info': {
                'name': self.info['name'] ,
                'video_dir': self.info['video_dir']
            },
            'data': {}
        }
        
        dic['data'] = data_json
        
        self.db.eye.insert_one(dic)
    
    def load_data_inf(self):
        
        data = self.db.eye.find_one({'info.video_dir': self.info['video_dir']})
        df_eye = pd.DataFrame.from_dict(data['data'])
        
        return df_eye
