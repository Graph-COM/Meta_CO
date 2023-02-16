import torch
from torch.optim import Adam
from math import ceil
import torch.nn
import torch
import pickle
from torch_geometric.data import DataListLoader, DataLoader
from random import shuffle
from torch_geometric.datasets import TUDataset
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gnn_model import vertex_MPNN, ErdosLoss_vertex
import argparse
import os
from utils import get_diracs

from pathlib import Path
import yaml
from dataset.rb200_train import RB200_train
from dataset.rb200_val import RB200_val
from tqdm import tqdm
from test import decode_vertex
#from dataset.collab_train import COLLAB
#from dataset.collab_test import COLLAB_test

loss_list = []

def train(net, train_loader, optimizer, epoch, device, receptive_field, penalty_coeff, totalretdict, batch_size, criterion):
    losses = 0
    weights = 0
    distances = 0
    for data in train_loader:
        optimizer.zero_grad(), 
        data = data.to(device)
        data_prime = data
        data_prime = get_diracs(data, 1, sparse = True, effective_volume_range=0.15, receptive_field = receptive_field)
        data_prime = data_prime.to(device)
        probs = net(data_prime.x, data_prime.edge_index, data_prime.batch, None, penalty_coeff)
        retdict = criterion(probs, data.edge_index, data.batch, penalty_coeff, device)
        data = data.to('cpu')
        losses = losses + retdict['loss'][0].item()
        weights = weights + retdict['Expected weight'][0].item()
        distances = distances + retdict['Expected distance'][0].item()
        if epoch > 2:
            retdict["loss"][0].backward()
            #reporter.report()
            #torch.nn.utils.clip_grad_norm_(net.parameters(),1)
            optimizer.step()
            del(retdict)
    losses = losses / (len(train_loader.dataset) / batch_size)
    weights = weights / (len(train_loader.dataset) / batch_size)
    distances = distances / (len(train_loader.dataset) / batch_size)
    del data_prime
    print(str(losses)+ ' ' + str(weights) + ' ' + str(distances))
    loss_list.append(losses)
    


def main():
    parser = argparse.ArgumentParser(description='this is the arg parser for max clique problem')
    parser.add_argument('--save_path', dest = 'save_path',default = './train_files/erdos/new_train/')
    parser.add_argument('--gpu', dest = 'gpu',default = '0')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    # prepare the dataset

    cfg = Path('./dataset/configs/config.yaml')
    cfg_dict = yaml.safe_load(cfg.open('r'))
    dataset = RB200_train(cfg_dict['train'])
    #import pdb; pdb.set_trace()
    batch_size = 96
    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    testset = RB200_val(cfg_dict['val'])
    test_loader = DataLoader(testset, 1, shuffle=False)

    #set up random seeds 
    torch.manual_seed(66)
    np.random.seed(2)   
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #hyper-parameters
    epochs = 5000
    learning_rate = 0.001
    numlayers = 4
    receptive_field = numlayers + 1
    penalty_coeff = 4
    hidden_1 = 64
    hidden_2 = 1


    net =  vertex_MPNN(dataset, numlayers, hidden_1, hidden_2 ,1)
    net.to(device).reset_parameters()
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=0.00000)
    criterion = ErdosLoss_vertex()
    # train the model
    lowest_score = 100
    lowest_loss = 10000
    for epoch in tqdm(range(epochs)):
        totalretdict = {}
        if epoch % 10 == 0:
            penalty_coeff = penalty_coeff + 0.
            print("Penalty_coefficient: ", penalty_coeff)
        #show currrent epoch and GPU utilizationss
        print('Epoch: ', epoch)
        #GPUtil.showUtilization()
        net.train()
        train(net, train_loader, optimizer, epoch, device, receptive_field, penalty_coeff, totalretdict, batch_size, criterion)
        if loss_list[epoch] < lowest_loss:
            lowest_loss = loss_list[epoch]
            model_path = args.save_path + 'best_loss.pth'
            torch.save(net.state_dict(), model_path)
        
        if epoch % 5 == 0:
            model_output = np.zeros(len(testset))
            model_output = model_output + 10000
            gt_output = []
            model_index = -1
            time_list = []
            for data in test_loader:
                model_index = model_index + 1
                for k in range(1):
                    # get k different data input
                    data_prime = get_diracs(data.to(device), 1, sparse = True, effective_volume_range=0.15, receptive_field = receptive_field)
                    data_prime = data_prime.to(device)
                    criterion = ErdosLoss_vertex()
                    probs = net(data_prime.x, data_prime.edge_index, data_prime.batch, None, penalty_coeff)
                    retdict = criterion(probs, data_prime.edge_index, data_prime.batch, penalty_coeff, device)
                    num_vertex = decode_vertex(data_prime,probs)
                    if num_vertex.item() < model_output[model_index]:
                        model_output[model_index] = num_vertex
                vertex_num = data.min_cover.item()
                gt_output.append(vertex_num)
                #print('model_index:'+str(model_index)+" gt:"+str(cliqno)+' model:'+str(model_output[model_index])+"time:"+str(time_per_data))
            ratios = [(model_output[i] - gt_output[i])/gt_output[i] for i in range(len(model_output))]
            print(f"Mean ratio: {(np.array(ratios)).mean()} +/-  {(np.array(ratios)).std()}")
            if (np.array(ratios)).mean() < lowest_score:
                lowest_score = (np.array(ratios)).mean()
                model_path = args.save_path + 'best_model'+str(epoch)+'.pth'
                torch.save(net.state_dict(), model_path)
                print("epoch:"+str(epoch)+", get best again")
        
        train_model_path = args.save_path + 'train_model.pth'
        torch.save(net.state_dict(), train_model_path)

    loss_list_path = args.save_path + 'loss_list.pkl'
    loss_list_file = open(loss_list_path,'wb')
    pickle.dump(loss_list, loss_list_file)
if __name__ == '__main__':
    main()