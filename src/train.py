import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from test import test
import gc


def train(net,train_loader,val_loader,params,test_loader1,test_loader2):

    init = params['init']
    device = params['device']
    criterion = params['criterion']
    optim = params['optim']
    num_epoch = params['num_epoch']
    lr = params['lr']
    lr_min = params['lr_min']
    scheduler_type = params['scheduler_type']
    weight_decay = params['weight_decay']
    best_model_path = params['best_model_path']

    net.to(device)

    def xavier():
        for module in net.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
    def kaming():
        for module in net.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)


    if init == 'xavier':
        net.apply(xavier)
    elif init == 'kaning':
        net.apply(kaming)


    if criterion == 'mse':
        criterion = nn.MSELoss(reduction='sum')
        #criterion = nn.MSELoss()
    elif criterion == 'ce':
        criterion = nn.CrossEntropyLoss().float()
    loss_function = criterion


    if optim == 'sgd':
        optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=weight_decay)
    elif optim == 'adam':
        optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=weight_decay)
    elif optim == 'adamW':
        optimizer = torch.optim.AdamW((param for param in net.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=0)


    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)








    train_loss = []
    val_loss = []
    best_value = 100
    best_val_metric = None
    best_net = None
    best_train_metrcs = None
    best_test_metrics1 = None
    best_test_metrics2 = None
    best_model_dict = None
    best_train_y_true = None
    best_train_y_pred = None
    best_val_y_true = None
    best_val_y_pred = None
    best_test_y_true1 = None
    best_test_y_pred1 = None
    best_test_y_true2 = None
    best_test_y_pred2 = None
    best_epoch_num = 0

    net.train()
    for id,e in enumerate(range(num_epoch)):
        for batch_data in tqdm(train_loader):
            pdbid,smi_encode,pro_seq_encode,poc_seq_encode,mask_poc_encode,affinity = batch_data
            poc_seq_encode = poc_seq_encode.to(device)
            smi = smi_encode
            smi = smi.to(device)
            smi_encode = smi_encode.to(device)
            pro_seq_encode = pro_seq_encode.to(device)
            mask_poc_encode = mask_poc_encode.to(device)
            y = affinity.to(device)
            y_hat = net(pkt = poc_seq_encode,smi=smi,seq_encode=pro_seq_encode, smi_encode=smi_encode,poc_encode=mask_poc_encode)
            loss = criterion(y_hat.view(-1), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        scheduler.step()



        train_metrics,train_y_true,train_y_pred = test(net,train_loader,loss_function,device)
        e_train_loss = train_metrics['loss']
        e_train_rmse = train_metrics['RMSE']
        train_loss.append(e_train_loss)
        print(f'epoch{id + 1}:train         loss={e_train_loss}')
        val_metrics,val_y_true,val_y_pred = test(net,val_loader,loss_function,device)
        e_val_loss = val_metrics['loss']
        e_val_rmse = val_metrics['RMSE']
        val_loss.append(e_val_loss)
        print(f'epoch{id + 1}:validation    loss={e_val_loss}')
        test_metrics, test_y_true, test_y_pred = test(net, test_loader1, loss_function, device)
        test_metrics2, test_y_true2, test_y_pred2 = test(net, test_loader2, loss_function, device)





        if(val_metrics['RMSE']<best_value):
            best_value = val_metrics['RMSE']
            best_epoch_num = id+1
            best_net= net

            best_train_metrcs = train_metrics
            best_train_y_true = train_y_true
            best_train_y_pred = train_y_pred
            best_val_metric = val_metrics
            best_val_y_true = val_y_true
            best_val_y_pred = val_y_pred
            best_test_metrics1 = test_metrics
            best_test_y_pred1 = test_y_pred
            best_test_y_true1 = test_y_true
            best_test_metrics2 = test_metrics2
            best_test_y_pred2 = test_y_pred2
            best_test_y_true2 = test_y_true2

            best_model_dict = net.state_dict()
            torch.save(best_model_dict, os.path.join(best_model_path, 'best_model.pth'))




    np.save(os.path.join(best_model_path,'train_loss.npy'),np.array(train_loss))
    np.save(os.path.join(best_model_path,'best_train_y_true.npy'),best_train_y_true)
    np.save(os.path.join(best_model_path, 'best_train_y_pred.npy'), best_train_y_pred)
    np.save(os.path.join(best_model_path, 'val_loss.npy'), np.array(val_loss))
    np.save(os.path.join(best_model_path, 'best_val_y_true.npy'), best_val_y_true)
    np.save(os.path.join(best_model_path, 'best_val_y_pred.npy'), best_val_y_pred)
    np.save(os.path.join(best_model_path, 'best_test_y_true1.npy'), best_test_y_true1)
    np.save(os.path.join(best_model_path, 'best_test_y_pred1.npy'), best_test_y_pred1)
    np.save(os.path.join(best_model_path, 'best_test_y_true2.npy'), best_test_y_true2)
    np.save(os.path.join(best_model_path, 'best_test_y_pred2.npy'), best_test_y_pred2)

    with open(os.path.join(best_model_path,'train_val_result.txt'),'w') as file:
        file.write(f'Best epoch at {best_epoch_num}:\n ')
        file.write('train_result:\n')
        for key, value in best_train_metrcs.items():
            file.write(f"{key}: {value}\n")
        file.write('\n\n')
        file.write('val_result:\n')
        for key, value in best_val_metric.items():
            file.write(f"{key}: {value}\n")
        file.write('\n\n')
        file.write('CASF2013_result:\n')
        for key, value in best_test_metrics1.items():
            file.write(f"{key}: {value}\n")
        file.write('\n\n')
        file.write('CASF2016_result:\n')
        for key, value in best_test_metrics2.items():
            file.write(f"{key}: {value}\n")
    
    with open(os.path.join(best_model_path,'params.txt'),'w') as file2:
        for key,value in params.items():
            file2.write(f'{key}: {value}\n')



    plt.plot(np.arange(1, num_epoch + 1), np.array(train_loss), label='train')
    plt.plot(np.arange(1, num_epoch + 1), np.array(val_loss), label='validation')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('LOSS', fontsize=20)
    plt.ylim(0,3)
    plt.legend(fontsize=18)
    plt.show()









