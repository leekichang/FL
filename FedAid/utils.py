import argparse

import os
import torch
import argparse
import numpy as np
import config as cfg
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from models import *

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parse_args():
    parser = argparse.ArgumentParser(description='FL PROJECT')
    parser.add_argument('--model', default = 'dnn', type = str,
                        choices=['cnn', 'dnn'])
    parser.add_argument('--dataset', default='femnist', type=str,
                        choices=['femnist'])
    parser.add_argument('--T_type', default=0, type=int)
    args = parser.parse_args(args=[])
    #args = parser.parse_args()
    return args

def args2dict(args:argparse.Namespace):
    args_dict = {'model':args.model,
                 'dataset':args.dataset,
                 'T_type':args.T_type,}
    return args_dict

def get_model(model_name:str):
    model_dict = {
        "cnn" : 'CNN()',
        "dnn" : 'Net()'
    }
    return eval(model_dict[model_name])

if __name__ == '__main__':
    args = parse_args()
    args_dict = args2dict(args)
    model = get_model(args_dict['model'])
    print(model)