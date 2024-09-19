from test import test
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from model.LS_Net import LSA_Module
from dataset import MyDataset

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
params = {
    'device': 'cpu',
    'criterion': 'mse',
    'batch_size':32 #
}

device = params['device']
batch_size = params['batch_size']
protein_seq_len = 868
pocket_seq_len = 66
smi_len = 147
dataset_name2016 = 'CASF2016'
dataset_name2013 = 'CASF2013'
test_set_path = '../data'
test_dataset = MyDataset(test_set_path, dataset_name2013, protein_seq_len, pocket_seq_len, smi_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_dataset2 = MyDataset(test_set_path, dataset_name2016, protein_seq_len, pocket_seq_len, smi_len)
test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False, drop_last=False)
loss_function = nn.MSELoss(reduction='sum')
net = LSA_Module()
best_m_p = '../best_model.pth'
net.load_state_dict(torch.load(best_m_p))
net.to(device)
test_metrics, test_y_true, test_y_pred = test(net, test_loader, loss_function, device)
test_metrics2, test_y_true2, test_y_pred2 = test(net, test_loader2, loss_function, device)
print('casf2013:')
print(test_metrics)
print("casf2016:")
print(test_metrics2)