import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 62)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def FedAvg(models:list, num_samples:list, layers:list=None, weighted:bool=True):
    assert len(models)==len(num_samples), "len(models)==len(numsamples)"
    
    average_model = copy.deepcopy(models[0])
    
    if layers==None:
        layers = average_model.keys()        
    
    if weighted:
        weights = np.array(num_samples)/np.sum(num_samples)
        print(f'weights: {weights}')
    else:
        weights = np.ones(np.shape(num_samples))/len(models)
    
    for layer in layers:
        average_model[layer] = weights[0] * average_model[layer]
        
        for idx, model in enumerate(models[1:]):
            for layer in layers:
                average_model[layer] += weights[idx+1] * model[layer]
    return copy.deepcopy(average_model)

if __name__ == '__main__':
    model = Net()
    models = [Net().state_dict(), Net().state_dict(), Net().state_dict(), Net().state_dict()]
    num_samples = [10, 20, 30, 40]
    avg_model = FedAvg(models, num_samples)
    