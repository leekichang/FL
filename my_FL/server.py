import torch
from torch.nn.functional import nll_loss
from torch.optim import SGD
from torch.utils.data import DataLoader
from models.simpleNet import Net

class Server():
    def __init__(self, dataset):
        self.model = Net()
        self.loss = nll_loss
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    def send_model(self):
        pass
    
    def global_test(self):
        pass
    
    def update_model(self):
        pass





