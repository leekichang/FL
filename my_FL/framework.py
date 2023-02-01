import os
import torch
import numpy as np
import config as cfg
import torch.nn as nn
import torch.optim as optim

from client import Client
from server import Server

from models.simpleNet import Net
import copy
from datamanager import *
import torch.utils.tensorboard as tb
import torch.multiprocessing as mp
from multiprocessing import Process

import torch.autograd as autograd

class FLFramework():
    def __init__(self, server:Server, clients:list, round:int=10):
        self.round = round
        self.model = Net()
        
        pass
    def save_model(self, epoch, tag='', ckpt:list=None):
        model_save_path = f'./checkpoints/{self.name}'
        if not os.path.isdir(model_save_path):
            os.mkdir(model_save_path)
        torch.save(self.model.state_dict(), f'{model_save_path}/{tag}_{epoch+1}.pth')         
    
    
if __name__ == '__main__':
    autograd.set_detect_anomaly(True)
    mp.set_start_method('spawn')

    PATH = cfg.DATAPATH['femnist']
    
    file_dict = get_files(PATH)
        
    TRAIN_DM = DataManager(file_dict['train'], is_train=True)
    TEST_DM = DataManager(file_dict['test'], is_train=False)
    DM_dict = {'train':TRAIN_DM,
               'test':TEST_DM}
    print("DATA READY")
    
    print(f"WORKING WITH {cfg.DEVICE}")
    
    server = Server(DM_dict)
    print("SERVER READY")
    
    server.setup()
    print('===== ROUND 0 =====\nServer Setup Complete!')
    
    for i in range(100):
        print(f'===== ROUND {i+1} START! =====\n')
        server.train_federated_model()
        server.global_test()
        print()
    
    # ROUND = 5
    # P = 50
    
    # for round in range(ROUND):
    #     print(f"======= ROUND {round+1} =======")
    #     # user selection
    #     selected_clients = server.select_users(n_participant=P)
        
    #     # user generation
    #     clients = []
    #     global_model = copy.deepcopy(server.distribute_model(selected_clients))
    #     for client in selected_clients:
    #         data_info = {'train':TRAIN_DM.get_user_info(client),\
    #                      'test':TEST_DM.get_user_info(client)}
    #         clients.append(Client(client_id=client, data_info=data_info, model=global_model[client]))
        
    #     # user local training
    #     procs = []
    #     for idx, c in enumerate(clients):
    #         proc = Process(target=c.local_train, args=())
    #         proc.start()
    #         procs.append(proc)
        
    #     for proc in procs:
    #         proc.join()

    #     # gather trained models
    #     train_result = []
    #     for client in clients:
    #         client.local_test()
    #         train_result.append(copy.deepcopy(client.upload_model()))

    #     # server side update
    #     # print(train_result)
    #     server.update_model(train_result)
    #     server.global_test()
        
        