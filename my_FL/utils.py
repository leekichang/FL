import argparse

import os
import cv2
import torch
import argparse
import framework
import numpy as np
import config as cfg
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

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
    #args = parser.parse_args(args=[])
    args = parser.parse_args()
    return args

def args2dict(args:argparse.Namespace):
    args_dict = {'model':args.model,
                 'dataset':None}
    return args_dict

if __name__ == '__main__':
    args = parse_args()