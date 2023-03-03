import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class DataManager():
    def __init__(self, files:list, is_train:bool=True, is_flat=True):
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

        if not is_flat:
            for idx in self.data:
                self.data[idx]['x'] = np.reshape(self.data[idx]['x'], (-1, 1, 28, 28))
            if not self.is_train:
                self.global_test_data['x'] = np.reshape(self.global_test_data['x'], (-1, 1, 28, 28))
        
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

    a = DataManager(files, is_train=False, is_flat=False)
    
    # dataset = FEMNIST(a.get_global_testset())
    
    # dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    
    # from tqdm import tqdm
    # for idx, batch in enumerate(tqdm(dataloader)):
    #     X, Y = batch
    #     print(X.shape, Y.shape)
    