import os
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATAPATH = {'femnist':'../leaf/data/femnist/data/', # TODO: dataset 별로 DATAPATH dictionary로 정의
            }

# TRAIN_FILES = [os.path.join(DATAPATH+'train', file) for file in os.listdir(PATH+'train') if file.endswith('.json')]
# TRAIN_FILES.sort()
# TEST_FILES = [os.path.join(DATAPATH+'test', file) for file in os.listdir(PATH+'test') if file.endswith('.json')]
# TEST_FILES.sort()


T_dict = {0:1, 1:0.1, 2:0.05, 3:0.01}