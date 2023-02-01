from pymongo import MongoClient
import json

class UserDB:
    def __init__(self):
        
        mongodb_URI = "mongodb+srv://heyi:20230214@hey-i.o4iunhl.mongodb.net/test"
        self.client = MongoClient(mongodb_URI)
        
        self.db = self.client['HEY-I']
        
    def save_data(self, name, date, video_dir): # name: name_phone, data: recording date, video_dir: GCS directory
        info = {}
        info['name'] = name
        info['date'] = date
        info['video_dir'] = video_dir
        
        # save user information
        
        self.db.user.insert_one(info)
            
            
class PoseDB(UserDB):
      
    def save_data(self, json_data, video_dir):
        # user info에 저장된 사용자의 정보를 가져온다.
        name = video_dir.split('/' )[1]
        dic = json.loads(json_data)
        
        dic['info']['name'] = name
        dic['info']['video_dir'] = video_dir
        
        self.db.pose.insert_one(dic)
        
    
    def load_data_inf(self, video_dir):
        
        data = self.db.pose.find_one({'info.video_dir': video_dir})

        return data
    
    def load_data_train(self):
        pass
        
        
class FaceDB(UserDB):
    
    def save_data(self, json_data, video_dir):
        # user info에 저장된 사용자의 정보를 가져온다.
        name = video_dir.split('/' )[1]
        dic = json.loads(json_data)
        
        dic['info']['name'] = name
        dic['info']['video_dir'] = video_dir
        dic['result'] = json_data
        
        self.db.face.insert_one(dic)
    
    def load_data_inf(self):
        pass
    
    def load_data_train(self):
        pass
        
class EyeDB(UserDB):
    
    def save_data(self, json_data, video_dir):
        # user info에 저장된 사용자의 정보를 가져온다.
        name = video_dir.split('/' )[1]
        dic = json.loads(json_data)
        
        dic['info']['name'] = name
        dic['info']['video_dir'] = video_dir
        dic['result'] = json_data
        
        self.db.face.insert_one(dic)
    
    def load_data_inf(self):
        pass
