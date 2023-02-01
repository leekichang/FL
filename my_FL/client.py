import warnings

import os
import copy
import torch
import config as cfg
import torch.nn as nn
from torch.optim import SGD
from models.simpleNet import Net
import sklearn.metrics as metrics
from torch.utils.data import DataLoader

warnings.filterwarnings(action='ignore')

from datamanager import *

class Client():
    def __init__(self, client_id:str, model:nn.Module, data_info:dict=None, device:str=cfg.DEVICE):
        self.id = client_id
        #self.cfg = cfg
        
        self.__model = None
        self.device = device
        
        self.train_info, self.test_info = data_info['train'], data_info['test'] # 함수화하기
        self.trainset, self.testset = FEMNIST(self.train_info), FEMNIST(self.test_info)
        
    @property
    def model(self):             
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model
    
    def __len__(self):
        return len(self.trainset)
    
    def setup(self):
        self.train_loader = DataLoader(self.trainset, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(self.testset, batch_size=16, shuffle=False)
        self.optimizer = SGD(self.model.parameters(), lr=0.01)     # TODO: utils.get_optimizer(cfg['optim']:str)
        self.criterion = nn.CrossEntropyLoss()                     # TODO: utils.get_loss(cfg['loss']:str)
        self.epochs = 10
    
    def local_train(self)->None:
        self.model.train()
        self.model.to(self.device)
        # TRAINING
        for epoch in range(self.epochs):
            for idx, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                X, Y = batch
                X, Y = X.to(self.device), Y.to(self.device)
                pred = self.model(X)
                loss = self.criterion(pred, Y)
                loss.backward()
                self.optimizer.step()
                if "cuda" in self.device : torch.cuda.empty_cache()
        # TESTING
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            loss_trace, result_pred, result_anno = [], [], []
            for idx, batch in enumerate(self.train_loader):
                X, Y = batch
                X, Y = X.to(self.device), Y.to(self.device)
                pred = self.model(X)
                loss = self.criterion(pred, Y)
                loss_trace.append(loss.to('cpu').detach().numpy())
                pred_np  = pred.to('cpu').detach().numpy()
                pred_np  = np.argmax(pred_np, axis=1).squeeze()
                Y_np     = Y.to('cpu').detach().numpy().reshape(-1, 1).squeeze()
                result_pred = np.hstack((result_pred, pred_np))
                result_anno = np.hstack((result_anno, Y_np))
                if "cuda" in self.device : torch.cuda.empty_cache()
            train_acc = metrics.accuracy_score(y_true=result_anno, y_pred=result_pred)
            train_loss = np.average(loss_trace)
            self.model.to('cpu')
        
        print(f'=== Client {self.id} Finished Training {len(self)} samples ===')
        print(f'client:{self.id} | Train Acc:{train_acc*100:.2f} | Train Loss:{train_loss:.4f}')
    
    def local_test(self):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            loss_trace, result_pred, result_anno = [], [], []
            for idx, batch in enumerate(self.test_loader):
                X, Y = batch
                X, Y = X.to(self.device), Y.to(self.device)
                pred = self.model(X)
                loss = self.criterion(pred, Y)
                loss_trace.append(loss.to('cpu').detach().numpy())
                pred_np  = pred.to('cpu').detach().numpy()
                pred_np  = np.argmax(pred_np, axis=1).squeeze()
                Y_np     = Y.to('cpu').detach().numpy().reshape(-1, 1).squeeze()
                result_pred = np.hstack((result_pred, pred_np))
                result_anno = np.hstack((result_anno, Y_np))
                
                if "cuda" in self.device : torch.cuda.empty_cache()
                
            test_acc = metrics.accuracy_score(y_true=result_anno, y_pred=result_pred)
            test_loss = np.average(loss_trace)
            print(f'client:{self.id} | Test Acc:{test_acc*100:.2f} | Test Loss:{test_loss:.4f}')
            self.model.to('cpu')

# class Client():
#     def __init__(self, client_id:str, model:nn.Module, data_info:dict=None):
#         self.id = client_id
#         #self.cfg = cfg
        
#         self.model = model
#         self.optimizer = SGD(self.model.parameters(), lr=0.00001)   # TODO: utils.get_optimizer(cfg['optim']:str)
#         self.criterion = nn.CrossEntropyLoss()                       # TODO: utils.get_loss(cfg['loss']:str)
#         self.epochs = 5
        
#         self.train_info, self.test_info = data_info['train'], data_info['test'] # 함수화하기
#         self.trainset, self.testset = FEMNIST(self.train_info), FEMNIST(self.test_info)
#         self.train_loader = DataLoader(self.trainset, batch_size=16, shuffle=True)
#         self.test_loader = DataLoader(self.testset, batch_size=16, shuffle=False)
    
#     def local_train(self)->None:
#         proc = os.getpid()
#         self.model.train()
#         for epoch in range(self.epochs):
#             for idx, batch in enumerate(self.train_loader):
#                 self.optimizer.zero_grad()
#                 X, Y = batch
#                 X, Y = X.to(cfg.DEVICE), Y.to(cfg.DEVICE)
#                 pred = self.model(X)
#                 loss = self.criterion(pred, Y)
#                 loss.backward()
#                 self.optimizer.step()
#         print(f'=== Process ID: {proc} | Client {self.id} Finished Training {self.trainset.__len__()} samples ===')
    
#     def upload_model(self):             # 주고 받는 모델은 state_dict로
#         return {'model':self.model.state_dict(), 'num_sample':self.trainset.__len__()}
    
#     def local_test(self):
#         self.model.eval()
#         with torch.no_grad():
#             loss_trace, result_pred, result_anno = [], [], []
#             for idx, batch in enumerate(self.test_loader):
#                 X, Y = batch
#                 X, Y = X.to(cfg.DEVICE), Y.to(cfg.DEVICE)
#                 pred = self.model(X)
#                 loss = self.criterion(pred, Y)
#                 loss_trace.append(loss.to('cpu').detach().numpy())
#                 pred_np  = pred.to('cpu').detach().numpy()
#                 pred_np  = np.argmax(pred_np, axis=1).squeeze()
#                 Y_np     = Y.to('cpu').detach().numpy().reshape(-1, 1).squeeze()
#                 result_pred = np.hstack((result_pred, pred_np))
#                 result_anno = np.hstack((result_anno, Y_np))
#             self.acc = metrics.accuracy_score(y_true=result_anno, y_pred=result_pred)
#             self.test_loss = np.average(loss_trace)
#             print(f'client:{self.id} | Acc:{self.acc*100:.2f} | Loss:{self.test_loss:.4f}')  

if __name__ == '__main__':
    PATH = cfg.DATAPATH['femnist']
    
    file_dict = get_files(PATH)
        
    TRAIN_DM = DataManager(file_dict['train'], is_train=True)
    TEST_DM = DataManager(file_dict['test'], is_train=False)
    data_info = {'train':TRAIN_DM.get_user_info('f1755_46'), 'test':TEST_DM.get_user_info('f1755_46')}
    
    client = Client(client_id=TRAIN_DM.users[0], data_info=data_info)
    client.local_train()
    client.local_test()
    
# # Perform Federated Learning on the client
# for round_num in range(NUM_ROUND):
#     # Perform local training on the client
#     for batch_num, (batch_data, batch_target) in enumerate(dataloader):
#         optimizer.zero_grad()
#         output = model(batch_data)
#         loss = loss_fn(output, batch_target)
#         loss.backward()
#         optimizer.step()

#     # Send the model to the server
#     model_params = model.state_dict()
#     send_model_to_server(model_params)

#     # Receive the updated model from the server
#     updated_model_params = receive_updated_model_from_server()
#     model.load_state_dict(updated_model_params)

# # Perform inference on the client
# inference_data = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
# inference_output = model(inference_data)
# print(inference_output)