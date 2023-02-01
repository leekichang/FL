import os
import copy
import torch
import numpy as np
import config as cfg
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.simpleNet import Net
from datamanager import *
from client import *
from Fed_Algorithms import FedAvg
import sklearn.metrics as metrics
from collections import OrderedDict
from multiprocessing import pool, cpu_count
from multiprocessing import Process

class Server():
    def __init__(self, DM_dict:dict, algorithm:str=None):
        self.train_DM = DM_dict['train']
        self.test_DM = DM_dict['test']
        
        self.clients = None
        self.device = cfg.DEVICE
        
        self.global_model = Net()
        
        self.criterion = nn.CrossEntropyLoss()              # TODO: utils.get_loss(cfg['loss']:str)
        
        self.Algorithm = FedAvg.FedAvg                      # FedAVG 같은 aggrrgation method 들어감 TODO: utils.get_algortihm() 작성
        self.received_models = None                         # Client.upload_model() 결과가 여기 들어감
        
        self.mp_flag = True

    def setup(self):
        self.clients = self.create_clients()
        self.data = FEMNIST(self.test_DM.global_test_data)
        self.dataloader = DataLoader(self.data, batch_size=256, shuffle=False)
        
        self.transmit_model()
        self.setup_clients()
        
        
    def create_clients(self, n_users:int=35):
        self.user_ids = self.test_DM.users
        self.user_ids = np.random.choice(self.user_ids, n_users, replace=False)
        clients = {}
        for user in self.user_ids:
            data_info = {'train':self.train_DM.data[user],\
                         'test':self.test_DM.data[user]}
            clients[user] = Client(client_id=user, model=self.global_model, data_info=data_info, device=self.device)
        return clients
    
    def setup_clients(self)->None:
        for k, client in tqdm(enumerate(self.clients), leave=False):
            self.clients[client].setup()
    
    def transmit_model(self, sampled_clients:list=None)->None:
        if sampled_clients == None:
            for client in tqdm(self.clients, leave=False):
                self.clients[client].model = copy.deepcopy(self.global_model)
        else:
            for client in tqdm(sampled_clients, leave=False):
                self.clients[client].model = copy.deepcopy(self.global_model)

        
    def sample_clients(self, n_participant:int=10)->np.array:
        assert n_participant <= len(self.user_ids), "Check 'n_participant <= len(self.clients)'"
        return np.random.choice(self.user_ids, n_participant, replace=False) # 입력된 수의 유저를 추출해서 반환

    def train_selected_clients(self, sampled_clients:list)->None:
        total_sample = 0
        for client in tqdm(sampled_clients, leave=False):
            self.clients[client].local_train()
            total_sample += len(self.clients[client])
        return total_sample
    
    def mp_train_selected_clients(self, procnum:int, client:str)->None:
        self.clients[client].local_train()
    
    def test_selected_models(self, sampled_clients):
        for client in sampled_clients:
            self.clients[client].local_test()

    def mp_test_selected_models(self, procnum:int, client:str):
        self.clients[client].local_test()
    
    def average_model(self, sampled_clients, coefficients):
        averaged_weights = OrderedDict()
        for it, client in tqdm(enumerate(sampled_clients), leave=False):
            local_weights = self.clients[client].model.state_dict()
            for key in self.global_model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.global_model.load_state_dict(averaged_weights)
    
    def update_model(self, train_result:dict, layers:list=None):
        self.received_models, num_samples = [], []
        for result in train_result:
            self.received_models.append(result['model'])
            num_samples.append(result['num_sample'])
        state = self.Algorithm(self.received_models, num_samples, layers)
        self.global_model.load_state_dict(state)

    def train_federated_model(self):
        sampled_clients = self.sample_clients()
        print(f"CLIENTS {sampled_clients} ARE SELECTED!\n")
        
        if self.mp_flag:
            print("TRAIN WITH MP!\n")
            procs = []
            selected_total_size = []
            for idx, c in enumerate(sampled_clients):
                selected_total_size.append(len(self.clients[c]))
                proc = Process(target=self.mp_train_selected_clients, args=(idx, c))
                proc.start()
                procs.append(proc)
            for proc in procs:
                proc.join()
            selected_total_size = sum(selected_total_size)
            # with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
            #     selected_total_size = workhorse.map(self.mp_train_selected_clients, sampled_clients)
            
        else:
            print("TRAIN WITH SP!\n")
            selected_total_size = self.train_selected_clients(sampled_clients)

        # if not self.mp_flag:
        #     print("TEST WITH MP!\n")
        #     procs = []
        #     for idx, c in enumerate(sampled_clients):
        #         proc = Process(target=self.mp_test_selected_models, args=(idx, c))
        #         proc.start()
        #         procs.append(proc)
        #     for proc in procs:
        #         proc.join()
        #     # with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
        #     #     workhorse.map(self.mp_test_selected_models, sampled_clients)
        # else:
        print("TEST WITH SP!\n")
        self.test_selected_models(sampled_clients)
        
        mixing_coefficients = [len(self.clients[client]) / selected_total_size for client in sampled_clients]
        
        self.average_model(sampled_clients, mixing_coefficients)
        
    def global_test(self):
        self.global_model.eval()
        self.global_model.to(self.device)
        
        with torch.no_grad():
            loss_trace, result_pred, result_anno = [], [], []
            for idx, batch in enumerate(self.dataloader):
                X, Y = batch
                X, Y = X.to(self.device), Y.to(self.device)
                pred = self.global_model(X)
                loss = self.criterion(pred, Y)
                loss_trace.append(loss.to('cpu').detach().numpy())
                pred_np  = pred.to('cpu').detach().numpy()
                pred_np  = np.argmax(pred_np, axis=1).squeeze()
                Y_np     = Y.to('cpu').detach().numpy().reshape(-1, 1).squeeze()
                result_pred = np.hstack((result_pred, pred_np))
                result_anno = np.hstack((result_anno, Y_np))
            self.acc = metrics.accuracy_score(y_true=result_anno, y_pred=result_pred)
            self.test_loss = np.average(loss_trace)
            print(f'Global Test Result | Acc:{self.acc*100:.2f}, Loss:{self.test_loss:.4f}')
            self.global_model.to('cpu')

# class Server():
#     def __init__(self, DM_dict:dict, algorithm:str=None):
#         self.train_DM = DM_dict['train']
#         self.test_DM = DM_dict['test']
        
#         self.global_model = Net()
        
#         self.criterion = nn.CrossEntropyLoss()              # TODO: utils.get_loss(cfg['loss']:str)
        
#         self.Algorithm = FedAvg.FedAvg                      # FedAVG 같은 aggrrgation method 들어감 TODO: utils.get_algortihm() 작성
#         self.received_models = None                         # Client.upload_model() 결과가 여기 들어감
#         self.global_test_dataset = FEMNIST(self.test_DM.get_global_testset())
#         self.test_loader = DataLoader(self.global_test_dataset, batch_size=256, shuffle=False)
    
#     def create_clients(self):
#         self.user_ids = self.test_DM.users
#         print(self.user_ids)
    
#     def select_users(self, n_participant:int)->np.array:
#         assert n_participant <= len(self.test_DM.users), "Check 'n_participant <= len(self.test_DM.users)'"
#         return np.random.choice(self.test_DM.users, n_participant, replace=False) # 입력된 수의 유저를 추출해서 반환

#     def update_model(self, train_result:dict, layers:list=None):
#         self.received_models, num_samples = [], []
#         for result in train_result:
#             self.received_models.append(result['model'])
#             num_samples.append(result['num_sample'])
#         state = self.Algorithm(self.received_models, num_samples, layers)
#         self.global_model.load_state_dict(state)

#     def distribute_model(self, participants:list)->dict:
#         models = {p:copy.deepcopy(self.global_model) for p in participants}
#         return models
    
#     def global_test(self):
#         self.global_model.to(cfg.DEVICE)
#         self.global_model.eval()
        
#         with torch.no_grad():
#             loss_trace, result_pred, result_anno = [], [], []
#             for idx, batch in enumerate(self.test_loader):
#                 X, Y = batch
#                 X, Y = X.to(cfg.DEVICE), Y.to(cfg.DEVICE)
#                 pred = self.global_model(X)
#                 loss = self.criterion(pred, Y)
#                 loss_trace.append(loss.to('cpu').detach().numpy())
#                 pred_np  = pred.to('cpu').detach().numpy()
#                 pred_np  = np.argmax(pred_np, axis=1).squeeze()
#                 Y_np     = Y.to('cpu').detach().numpy().reshape(-1, 1).squeeze()
#                 result_pred = np.hstack((result_pred, pred_np))
#                 result_anno = np.hstack((result_anno, Y_np))
#             self.acc = metrics.accuracy_score(y_true=result_anno, y_pred=result_pred)
#             self.test_loss = np.average(loss_trace)
#             print(f'Global Test Result | Acc:{self.acc*100:.2f}, Loss:{self.test_loss:.4f}')  

if __name__ == '__main__':
    PATH = '../leaf/data/femnist/data/test'
    files = [os.path.join(PATH, file) for file in os.listdir(PATH) if file.endswith('.json')]
    files.sort()
    DM = DataManager(files, is_train=False)
    tmp = {'train':None, 'test':DM}
    server = Server(DM)
    server.create_clients()
    # server.select_users(10)
    
