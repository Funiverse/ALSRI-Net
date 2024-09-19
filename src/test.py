from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error # 回归
from scipy import stats # 相关系数
import metrics
import torch
import numpy as np
from tqdm import tqdm

def test(model, test_loader, loss_function, device):
    test_loss = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for pdbid, smi_encode, pro_seq_encode, poc_seq_encode, mask_poc_encode, affinity in test_loader:
            poc_seq_encode = poc_seq_encode.to(device)
            smi = smi_encode.to(device)
            smi_encode = smi_encode.to(device)
            pro_seq_encode = pro_seq_encode.to(device)
            mask_poc_encode = mask_poc_encode.to(device)
            y = affinity.to(device)
            y_hat = model(pkt = poc_seq_encode,smi=smi,seq_encode=pro_seq_encode, smi_encode=smi_encode,poc_encode=mask_poc_encode)

            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()

            y_pred.append(y_hat.cpu().numpy().reshape(-1))
            y_true.append(y.cpu().numpy().reshape(-1))

    y_true = np.concatenate(y_true).reshape(-1)
    y_pred = np.concatenate(y_pred).reshape(-1)

    test_loss /= len(test_loader.dataset)

    evaluation = {
        'loss': test_loss,
        'c_index': metrics.c_index(y_true, y_pred),
        'RMSE': metrics.RMSE(y_true, y_pred),
        'MAE': metrics.MAE(y_true, y_pred),
        'SD': metrics.SD(y_true, y_pred),
        'CORR': metrics.CORR(y_true, y_pred),
    }
    return evaluation,y_true,y_pred
