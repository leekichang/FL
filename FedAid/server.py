import os
import copy
import utils
import torch
import numpy as np
from models import *
from client import *
import config as cfg
import torch.nn as nn
from tqdm import tqdm
from datamanager import *
from datetime import datetime
from models.simpleNet import Net
import sklearn.metrics as metrics
from collections import OrderedDict
from multiprocessing import Process
from torch.utils.data import DataLoader
from multiprocessing import pool, cpu_count

import torch.utils.tensorboard as tb

class Server():
    def __init__(self, DM_dict:dict, args_dict:dict, algorithm:str=None):
        self.train_DM       = DM_dict['train']
        self.test_DM        = DM_dict['test']
        self.clients        = None
        self.args_dict      = args_dict
        self.device         = cfg.DEVICE
        self.global_model   = utils.get_model(args_dict['model'])
        self.criterion      = nn.CrossEntropyLoss()              # TODO: utils.get_loss(cfg['loss']:str)
        self.mp_flag        = False
        self.round          = 0
        self.rounds         = 10
        self.T_type         = args_dict['T_type']
        self.T              = []
        self.TB_WRITER      = tb.SummaryWriter(f'./tensorboard/{str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}_{args_dict["model"]}_{args_dict["dataset"]}_{cfg.T_dict[self.T_type]}')
        
    def setup(self):
        self.clients    = self.create_clients()
        self.data       = FEMNIST(self.test_DM.global_test_data)
        self.dataloader = DataLoader(self.data, batch_size=256, shuffle=False)
        self.transmit_model()
        self.setup_clients()
        
    def create_clients(self, n_users:int=35):
        self.user_ids   = self.test_DM.users
        self.user_ids   = np.random.choice(self.user_ids, n_users, replace=False)
        for i in range(n_users):
            self.T.append(cfg.T_dict[self.T_type])
        self.mean_T = np.average(self.T)
        # TODO: T 종류별로 미리 정해두기
        
        clients         = {}
        for idx, user in enumerate(self.user_ids): 
            data_info           = {'train':self.train_DM.data[user],\
                                   'test' : self.test_DM.data[user]}
            
            clients[user]       = Client(client_id=user, model=self.global_model, data_info=data_info, device=self.device)
            clients[user].model = copy.deepcopy(self.global_model)
            clients[user].T     = self.T[idx]
        return clients
    
    
    def setup_clients(self)->None:
        for k, client in tqdm(enumerate(self.clients), leave=False):
            self.clients[client].setup()
    
    
    def transmit_model(self, sampled_clients:list=None)->None:
        if sampled_clients == None:   
            for client in tqdm(self.clients, leave=False):
                self.clients[client].model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))
        else:
            for client in tqdm(sampled_clients, leave=False):
                self.clients[client].model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))


    def sample_clients(self, n_participant:int=10)->np.array:
        assert n_participant <= len(self.user_ids), "Check 'n_participant <= len(self.clients)'"
        selected_clients = np.random.choice(self.user_ids, n_participant, replace=False) 
        for client in selected_clients:
            self.clients[client].selected_rounds.append(self.round)
        return selected_clients

    
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
        self.round += 1
        sampled_clients = self.sample_clients()
        print(f"CLIENTS {sampled_clients} ARE SELECTED!\n")
        
        if self.mp_flag:
            print("TRAIN WITH MP!\n")
            procs = []
            selected_total_size = []
            for idx, c in enumerate(sampled_clients):
                selected_total_size.append(len(self.clients[c]))
                proc = Process(target=self.mp_train_selected_clients, args=(idx, c))    # with Multi-Process
                proc.start()
                procs.append(proc)
            for proc in procs:
                proc.join()
            selected_total_size = sum(selected_total_size)
            # with pool.ThreadPool(processes=cpu_count() - 1) as workhorse: # with Threading
            #     selected_total_size = workhorse.map(self.mp_train_selected_clients, sampled_clients)
            
        else:
            print("TRAIN WITH SP!\n")
            selected_total_size = self.train_selected_clients(sampled_clients)

        print("TEST WITH SP!\n")
        self.test_selected_models(sampled_clients)
                
        mixing_coefficients = [len(self.clients[client]) / selected_total_size for client in sampled_clients]
        
        self.average_model(sampled_clients, mixing_coefficients)
        self.transmit_model()
        
        
    def global_test(self):
        self.global_model.eval()
        self.global_model.to(self.device)
        
        with torch.no_grad():
            loss_trace, result_pred, result_anno = [], [], []
            for idx, batch in enumerate(self.dataloader):
                X, Y = batch
                X, Y = X.to(self.device), Y.to(self.device)
                pred = self.global_model(X) / self.mean_T
                loss = self.criterion(pred, Y)
                loss_trace.append(loss.to('cpu').detach().numpy())
                pred_np  = pred.to('cpu').detach().numpy()
                pred_np  = np.argmax(pred_np, axis=1).squeeze()
                Y_np     = Y.to('cpu').detach().numpy().reshape(-1, 1).squeeze()
                result_pred = np.hstack((result_pred, pred_np))
                result_anno = np.hstack((result_anno, Y_np))
            self.acc = metrics.accuracy_score(y_true=result_anno, y_pred=result_pred)
            self.test_loss = np.average(loss_trace)
            self.TB_WRITER.add_scalar(f'Global Test Accuracy', self.acc      , self.round)
            self.TB_WRITER.add_scalar(f'Global Test Loss'    , self.test_loss, self.round)
            print(f'Global Test Result | Acc:{self.acc*100:.2f}, Loss:{self.test_loss:.4f}')
            self.global_model.to('cpu')
    
    
    def test_all_client(self):
        with torch.no_grad():
            for c in self.clients:
                self.clients[c].local_test()
    
    
    def fit(self):
        pass
    
    
    def save_model(self, round=100, tag='', ckpt:list=None):
        model_save_path = f'./checkpoints/{self.args_dict["model"]}_{self.args_dict["dataset"]}_{cfg.T_dict[self.T_type]}'
        
        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)
        torch.save(self.global_model.state_dict(), f'{model_save_path}/{tag}_{round}.pth')   


if __name__ == '__main__':
    PATH = '../leaf/data/femnist/data/test'
    files = [os.path.join(PATH, file) for file in os.listdir(PATH) if file.endswith('.json')]
    files.sort()
    DM = DataManager(files, is_train=False)
    tmp = {'train':None, 'test':DM}
    server = Server(DM)
    server.create_clients()
    # server.select_users(10)
    
