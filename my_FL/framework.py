import os
import torch
import numpy as np
import config as cfg
import torch.nn as nn
import torch.optim as optim

import utils
from client import Client
from server import Server

from models.simpleNet import Net
import copy
from datamanager import *
import torch.utils.tensorboard as tb
import torch.multiprocessing as mp
from multiprocessing import Process

import torch.autograd as autograd
    
if __name__ == '__main__':
    autograd.set_detect_anomaly(True)
    mp.set_start_method('spawn')
    
    args = utils.parse_args()
    args_dict = utils.args2dict(args)
    
    PATH = cfg.DATAPATH['femnist']
        
    file_dict = get_files(PATH)
        
    TRAIN_DM = DataManager(file_dict['train'], is_train=True, is_flat=args.model=='dnn')
    TEST_DM = DataManager(file_dict['test'], is_train=False, is_flat=args.model=='dnn')
    DM_dict = {'train':TRAIN_DM,
            'test':TEST_DM}
    print("DATA READY")
        
    print(f"WORKING WITH {cfg.DEVICE}")
    
    server = Server(DM_dict, args_dict)
    print("SERVER READY")
        
    server.setup()
    print('===== ROUND 0 =====\nServer Setup Complete!')
    server.save_model(round=0)
    for i in range(100):
        print(f'===== ROUND {i+1} START! =====\n')
        server.train_federated_model()
        server.global_test()
        if (i+1) % 10 == 0:
            server.save_model(round=i+1)
        