import os
import random
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from train import train
from sklearn.model_selection import KFold
from model.LS_Net import LSA_Module
from dataset import MyDataset
from test import test





seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)

# 1.设置参数
curret_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_path = f'../saved_models/{curret_time}'
os.makedirs(folder_path)
params = {
    'init': 'kaming',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'criterion': 'mse',
    'optim': 'adamW',
    'lr': 1e-3,
    'lr_min': 1e-9,
    'scheduler_type': 'cosine',
    'weight_decay': 0,
    'best_model_path': folder_path,
    'num_epoch': 60,
    'batch_size':32
}
print('Training on:', params['device'])
device = params['device']
batch_size = params['batch_size']
protein_seq_len = 868
pocket_seq_len = 66
smi_len = 147

k_fold = 0


net = LSA_Module()



if k_fold == 0:
    print('Using given dataset')
    train_set_path = '../data'
    val_set_path = '../data'
    train_dataet = MyDataset(train_set_path,'training_1000v3',protein_seq_len,pocket_seq_len,smi_len)
    val_dataset = MyDataset(val_set_path, 'validation_1000v3',protein_seq_len,pocket_seq_len,smi_len)
    train_loader = DataLoader(train_dataet, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_set_path = '../data'
    dataset_name2013 = 'CASF2013'
    test_dataset = MyDataset(test_set_path, dataset_name2013, protein_seq_len, pocket_seq_len, smi_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    dataset_name2016 = 'CASF2016'
    test_dataset2 = MyDataset(test_set_path, dataset_name2016, protein_seq_len, pocket_seq_len, smi_len)
    test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=True, drop_last=False)

    train(net, train_loader, val_loader, params,test_loader,test_loader2)

else :
    print(f'{k_fold} fold cross validation')
