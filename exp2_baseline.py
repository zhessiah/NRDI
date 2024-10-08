from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from modules import *
import netrd

parser = argparse.ArgumentParser()
parser.add_argument('--graph', type=str, default='ws', help='ba,ws,rg,grid')
args = parser.parse_args()





def accuracy_score(y_true, y_pred):
    y_pred=y_pred.bool()
    y_true=y_true.bool()
    total = len(y_true)
    correct = torch.sum(y_pred == y_true)
    accuracy = correct / total
    return accuracy


for i in range(10,20):
    args.num_atoms = i
    train_loader = load_exp2_data(49, args.num_atoms, args.graph, 256, 10000, 8)

    acc_corr=[]
    acc_mi=[]
    acc_parcorr=[]
    off_diag_idx = np.ravel_multi_index(np.where(np.ones((args.num_atoms, args.num_atoms)) - np.eye(args.num_atoms)),[args.num_atoms, args.num_atoms])
    for batch_idx, (data, relations) in enumerate(train_loader):
        #relations [num_sims, num_atoms*num_atoms]
        # data  torch.Size([128, 5, 49, 4])
        # TS L,N
        data=data.sum(-1)# data  torch.Size([128, 5, 49])
        corr = netrd.reconstruction.CorrelationMatrix()
        mi = netrd.reconstruction.MutualInformationMatrix()
        parcorr = netrd.reconstruction.PartialCorrelationMatrix()
        avg_k = 4
        for i in range(data.shape[0]):
            TS=data[i].numpy()
            trueG=relations[i]
            corr_G=corr.fit(TS, threshold_type='degree', avg_k=avg_k)
            mi_G=mi.fit(TS, threshold_type='degree', avg_k=avg_k) #, threshold_type='degree', avg_k=avg_k
            parcorr_G=parcorr.fit(TS, threshold_type='degree', avg_k=avg_k)

            corr_G=torch.FloatTensor(nx.to_numpy_array(corr_G).reshape(-1))[off_diag_idx]
            mi_G=torch.FloatTensor(nx.to_numpy_array(mi_G).reshape(-1))[off_diag_idx]
            parcorr_G=torch.FloatTensor(nx.to_numpy_array(parcorr_G).reshape(-1))[off_diag_idx]

            acc_c = accuracy_score(corr_G, trueG)
            acc_m = accuracy_score(mi_G, trueG)
            acc_p = accuracy_score(parcorr_G, trueG)
            acc_corr.append(acc_c)
            acc_mi.append(acc_m)
            acc_parcorr.append(acc_p)
    print(args.graph,args.num_atoms)
    print('acc_corr: {:.10f}'.format(np.mean(acc_corr)))
    print('acc_mi: {:.10f}'.format(np.mean(acc_mi)))
    print('acc_parcorr: {:.10f}'.format(np.mean(acc_parcorr)))








