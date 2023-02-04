from pymongo import MongoClient
import json
import pandas as pd
import numpy as np

class UserDB:
    def __init__(self, name, num, date, video_dir): # name: name_phone, data: recording date, video_dir: GCS directory
        
        mongodb_URI = "mongodb+srv://heyi:20230214@hey-i.o4iunhl.mongodb.net/test"
        self.client = MongoClient(mongodb_URI)
        
        self.db = self.client['HEY-I']
        self.info = {}
        self.info['name'] = name
        self.info['num'] = num
        self.info['date'] = date
        self.info['video_dir'] = video_dir
        
    def save_data(self): 
        # save user information 
        self.db.user.insert_one(self.info)
            
            
class PoseDB(UserDB):
      
    def save_data(self, json_data):
        # user info에 저장된 사용자의 정보를 가져온다.
        
        dic = {
            'info': {
                'name': self.info['name'] ,
                'video_dir': self.info['video_dir']
            },
            'data': _ 
        }
        
        dic['data'] = json.loads(json_data)
        
        self.db.pose.insert_many(dic)
        

    def load_data_inf(self, ):
        
        data = self.db.pose.find_one({'info.video_dir': self.info['video_dir']})
        df_db = pd.DataFrame.from_dict(data['data'])
        
        return df_db
    
    def load_data_train(self):
        pass
        
        
class FaceDB(UserDB):
    
    def save_data(self, json_data):
        # user info에 저장된 사용자의 정보를 가져온다.
        
        dic = {
            'info': {
                'name': self.info['name'] ,
                'video_dir': self.info['video_dir']
            },
            'data': _ 
        }
        
        dic['data'] = json.loads(json_data)
        
        self.db.face.insert_one(dic)
    
    def load_data_inf(self):
        data = self.db.pose.find_one({'info.video_dir': self.info['video_dir']})
        df_db = pd.DataFrame.from_dict(data['data'])
        
        return df_db
    
    def load_data_train(self):
        pass
        
class EyeDB(UserDB):
    
    def save_data(self, json_data):
        # user info에 저장된 사용자의 정보를 가져온다.
        dic = {
            'info': {
                'name': self.info['name'] ,
                'video_dir': self.info['video_dir']
            },
            'data': _ 
        }
        
        dic['data'] = json.loads(json_data)
        
        self.db.face.insert_one(dic)
    
    def load_data_inf(self):
        
        data = self.db.pose.find_one({'info.video_dir': self.info['video_dir']})
        df_db = pd.DataFrame.from_dict(data['data'])
        
        return df_db
