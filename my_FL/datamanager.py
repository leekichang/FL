import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class DataManager():
    def __init__(self, files:list, is_train:bool=True):
        self.files = files
        self.is_train = is_train
        self.users, self.data = [], {}
        if not self.is_train:
            self.global_test_data = {'x':[], 'y':[]}
        
        for idx, file in enumerate(self.files):
            idx = str(idx)
            self.data[idx] = {'x':[], 'y':[]}
            with open(file) as f:
                data = json.load(f)
                self.users.append(idx)

                for user in data['users']:              # 각 유저의 data 저장
                    self.data[idx]['x'] = self.data[idx]['x'] + data['user_data'][user]['x']
                    self.data[idx]['y'] = self.data[idx]['y'] + data['user_data'][user]['y']
                    
                if not self.is_train:                   # test dataset인 경우 global evaluation 위해서 모든 데이터셋 저장
                    for user in data['users']:
                        self.global_test_data['x'] = self.global_test_data['x'] + data['user_data'][user]['x']
                        self.global_test_data['y'] = self.global_test_data['y'] + data['user_data'][user]['y']

class FEMNIST(Dataset):
    def __init__(self, data:dict):
        self.data = data
        self.data['x'] = np.array(data['x'])
        self.data['y'] = np.array(data['y'])
        
    def __getitem__(self, idx):
        #self.X = torch.tensor(self.data['x'][idx,:].reshape(1, 28, 28)).float()
        self.X = torch.tensor(self.data['x'][idx,:]).float()
        self.Y = torch.tensor(self.data['y'][idx]).long()
        return self.X, self.Y

    def __len__(self):
        return len(self.data['y'])
        
# class DataManager():
#     def __init__(self, files:list, is_train:bool=True):
#         self.files = files
#         self.is_train = is_train
#         self.users, self.data, self.users_group = [], {}, {}
        
#         if not self.is_train:
#             self.global_test_data = {'x':[], 'y':[]}
        
#         for file in self.files:
#             with open(file) as f:
#                 data = json.load(f)
#                 self.users = self.users+data['users']
#                 for user in data['users']:              # 각 유저가 어떤 file에서 왔는지 저장
#                     self.users_group[user] = int(file.split('_')[2]) # 일단 file 번호 기준으로 저장함
                    
#                 for user in data['users']:              # 각 유저의 data 저장
#                     self.data[user] = data['user_data'][user]
#                     self.data[user]['x'] = np.array(data['user_data'][user]['x'])
#                     self.data[user]['y'] = np.array(data['user_data'][user]['y'])

#                 if not self.is_train:                   # test dataset인 경우 global evaluation 위해서 모든 데이터셋 저장
#                     if len(self.global_test_data['y']) == 0:
#                         self.global_test_data['x'] = data['user_data'][data['users'][0]]['x']
#                         self.global_test_data['y'] = data['user_data'][data['users'][0]]['y']
#                     for user in data['users'][1:]:
#                         self.global_test_data['x'] = np.vstack([self.global_test_data['x'], data['user_data'][user]['x']])
#                         self.global_test_data['y'] = np.hstack([self.global_test_data['y'], data['user_data'][user]['y']])
        
#     def get_user_info(self, user:str=None)->dict:
#         return {'data':self.data[user], 'group':self.users_group[user]}
    
#     def get_global_testset(self)->dict:
#         return {'data':self.global_test_data, 'group':None}

# class FEMNIST(Dataset):
#     def __init__(self, data_info:dict):
#         self.data = data_info['data']
#         self.group = data_info['group']
        
#     def __getitem__(self, idx):
#         #self.X = torch.tensor(self.data['x'][idx,:].reshape(1, 28, 28)).float()
#         self.X = torch.tensor(self.data['x'][idx,:]).float()
#         self.Y = torch.tensor(self.data['y'][idx]).long()
#         return self.X, self.Y

#     def __len__(self):
#         return len(self.data['y'])

def get_files(PATH:str)->dict:
    file_dict = {'train':None, 'test':None}
    
    files = [os.path.join(PATH+'/train', file) \
        for file in os.listdir(PATH+'/train') if file.endswith('.json')]
    files.sort()
    file_dict['train'] = files
    
    files = [os.path.join(PATH+'/test', file) \
        for file in os.listdir(PATH+'/test') if file.endswith('.json')]
    files.sort()
    file_dict['test'] = files
    return file_dict

if __name__ == '__main__':
    PATH = '../leaf/data/femnist/data/test'
    files = [os.path.join(PATH, file) for file in os.listdir(PATH) if file.endswith('.json')]
    files.sort()

    a = DataManager(files, is_train=False)
    
    dataset = FEMNIST(a.get_global_testset())
    
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    
    from tqdm import tqdm
    for idx, batch in enumerate(tqdm(dataloader)):
        X, Y = batch
        print(X.shape, Y.shape)
    
    
    #print(a.get_user_info([a.users[0]]))

# for idx, file in enumerate(files):
#     with open(file) as f:
#         data = json.load(f)
#         print(f'File Index: {idx}')
#         print(len(data['users']))
#         print(data.keys())
#         break
        
        #print()