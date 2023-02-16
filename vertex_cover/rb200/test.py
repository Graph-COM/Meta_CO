import torch
import time
from math import ceil
import torch.nn
import torch
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops, segregate_self_loops, remove_isolated_nodes, contains_isolated_nodes, add_remaining_self_loops
from utils import get_diracs
import scipy
import scipy.io
from matplotlib.lines import Line2D
import GPUtil
import pickle
from torch_geometric.data import DataListLoader, DataLoader
from random import shuffle
from torch_geometric.datasets import TUDataset
import visdom 
from visdom import Visdom 
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from gnn_model import vertex_MPNN, ErdosLoss_vertex
from torch_geometric.nn.norm.graph_size_norm import GraphSizeNorm
import argparse
import os
from dataset.rb200hard_test import RB200hard_test
from pathlib import Path
import yaml


def decode_vertex(data, probs):
    edge_index = data.edge_index
    no_loop_index,_ = remove_self_loops(edge_index)  
    row, col = no_loop_index
    probs = probs.squeeze()
    for i in range(data.x.shape[0]):
        tmp_0 = probs.clone()
        tmp_1 = probs.clone()
        tmp_0[i] = 0
        tmp_1[i] = 1
        tmp_0_row = tmp_0[row]
        tmp_0_col = tmp_0[col]
        tmp_1_row = tmp_1[row]
        tmp_1_col = tmp_1[col]
        weight_0 = tmp_0.sum()
        weight_1 = tmp_1.sum()
        penalty_0 =  ((1 - tmp_0_row) * (1 - tmp_0_col)).sum()
        penalty_1 =  ((1 - tmp_1_row) * (1 - tmp_1_col)).sum()
        #import pdb; pdb.set_trace()
        if weight_0 + penalty_0 < weight_1 + penalty_1:
            probs = tmp_0.clone()
        else:
            probs = tmp_1.clone()
    return probs.sum()

def decode_vertex_fast(data, probs):
    edge_index = data.edge_index
    no_loop_index,_ = remove_self_loops(edge_index)  
    row, col = no_loop_index
    probs = probs.squeeze()
    for i in range(data.x.shape[0]):
        id = (row == i).nonzero(as_tuple=False)
        neighbor_of_i = col[id]
        penalty = torch.sum(1-probs[neighbor_of_i])
        if penalty < 0.5:
            probs[i] = 0
        else:
            probs[i] = 1
    probs_row = probs[row]
    probs_col = probs[col]
    penalty = ((1 - probs_row) * (1 - probs_col)).sum()
    #print(probs)
    if penalty > 0:
        print("penalty infer larger than 0: "+str(penalty))
    return probs.sum()


def main():
    parser = argparse.ArgumentParser(description='this is the arg parser for max clique problem')
    parser.add_argument('--save_path', dest = 'save_path',default = 'train_files/max_clique/new_train/')
    parser.add_argument('--gpu', dest = 'gpu',default = '0')
    args = parser.parse_args()

    
    # prepare the dataset
    cfg = Path('./dataset/configs/config.yaml')
    cfg_dict = yaml.safe_load(cfg.open('r'))
    testdata = RB200hard_test(cfg_dict['test'])
    dataset = testdata
    batch_size = 1
    test_loader = DataLoader(testdata, batch_size, shuffle=False)
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    #set up random seeds 
    torch.manual_seed(66)
    np.random.seed(2)   
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #hyper-parameters
    numlayers = 4
    receptive_field = numlayers + 1
    penalty_coeff = 4.
    hidden_1 = 64
    hidden_2 = 1
    
    net =  vertex_MPNN(dataset, numlayers, hidden_1, hidden_2 ,1)
    state_dict = torch.load('./train_files/maml/demo/best_model.pth', map_location = torch.device('cpu'))
    
    
    
    net.train()
    #net.eval()
    model_output = np.zeros(len(testdata))
    model_output = model_output + 10000
    gt_output = []
    model_index = -1
    time_list = []
    loss_list = []
    weight_list = []
    distance_list = []
    criterion = ErdosLoss_vertex()
    for data in test_loader:
        model_index = model_index + 1
        time_per_data = 0
        for k in range(8):
            # get k different data input
            data_prime = get_diracs(data.to(device), 1, sparse = True, effective_volume_range=0.15, receptive_field = receptive_field)
            data_prime = data_prime.to(device)
            net.reset_parameters()
            net.load_state_dict(state_dict)
            net.to(device)
            start_t = time.time()
            probs = net(data_prime.x, data_prime.edge_index, data_prime.batch, None, penalty_coeff)
            num_vertex = decode_vertex_fast(data_prime,probs)
            if num_vertex.item() < model_output[model_index]:
                model_output[model_index] = num_vertex
            end_t = time.time()
            retdict = criterion(probs, data_prime.edge_index, data_prime.batch, penalty_coeff, device)
            losses = retdict['loss'][0].item()
            weights = retdict['Expected weight'][0].item()
            distances = retdict['Expected distance'][0].item()
            loss_list.append(losses)
            weight_list.append(weights)
            distance_list.append(distances)
            time_per_data = time_per_data + end_t - start_t
            
        time_list.append(time_per_data)
        vertex_gt = data.min_cover
        gt_output.append(vertex_gt.item())
        print('model_index:'+str(model_index)+" gt:"+str(vertex_gt.item())+' model:'+str(model_output[model_index])+" all_node:"+str(data.x.shape[0])+" time:"+str(time_per_data))
    #import pdb; pdb.set_trace()
    
    ratios = [(model_output[i]-gt_output[i])/gt_output[i] for i in range(len(model_output))]
    diff_abs = [abs(model_output[i]-gt_output[i]) for i in range(len(model_output))]
    
    #ratios = ratios.numpy()
    
    print(f"Mean ratio: {(np.array(ratios)).mean()} +/-  {(np.array(ratios)).std()}")
    print(f"Mean diff abs: {(np.array(diff_abs)).mean()} +/-  {(np.array(diff_abs)).std()}")

    print('loss:'+str(np.mean(loss_list))+'weight:'+str(np.mean(weight_list))+'distance:'+str(np.mean(distance_list)))
    print('avg_time:')
    print(np.mean(time_list))

if __name__ == '__main__':
    main()