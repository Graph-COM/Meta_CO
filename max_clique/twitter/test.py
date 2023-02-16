import torch
import time
from math import ceil
import torch.nn
import torch
from utils import get_diracs
import scipy
import scipy.io
from matplotlib.lines import Line2D
import pickle
from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel
from torch_geometric.data import DataListLoader, DataLoader
from random import shuffle
from torch_geometric.datasets import TUDataset
import visdom 
from visdom import Visdom 
import numpy as np
import matplotlib.pyplot as plt
from  utils import solve_gurobi_maxclique
import gurobipy as gp
from gurobipy import GRB
from gnn_model import clique_MPNN, ErdosLoss_clique
from torch_geometric.nn.norm.graph_size_norm import GraphSizeNorm
from utils import decode_clique_final, decode_clique_final_speed
import argparse
import os
from pathlib import Path
import yaml
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from dataset.twitter_test import TWITTER_test

    


def main():
    parser = argparse.ArgumentParser(description='this is the arg parser for max clique problem')
    parser.add_argument('--save_path', dest = 'save_path',default = 'train_files/max_clique/new_train/')
    parser.add_argument('--gpu', dest = 'gpu',default = '0')
    args = parser.parse_args()

    
    # prepare the dataset
    cfg = Path('./dataset/configs/config.yaml')
    cfg_dict = yaml.safe_load(cfg.open('r'))
    testdata = TWITTER_test(cfg_dict['test'])
    dataset = testdata
    batch_size = 1
    test_loader = DataLoader(testdata, batch_size, shuffle=False)
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
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
    
    net =  clique_MPNN(dataset, numlayers, hidden_1, hidden_2 ,1)

    state_dict = torch.load('./train_files/maml/demo/best_model.pth',map_location = torch.device('cpu'))


    net.train()
    model_output = np.zeros(len(testdata))
    gt_output = []
    model_index = -1
    time_list = []
    for data in test_loader:
        model_index = model_index + 1
        time_per_data = 0
        start_t = time.time()
        for k in range(8):
            # get k different data input
            data_prime = get_diracs(data.to(device), 1, sparse = True, effective_volume_range=0.15, receptive_field = receptive_field)
            data_prime = data_prime.to(device)
            net.reset_parameters()
            net.load_state_dict(state_dict)
            net.to(device)
            criterion = ErdosLoss_clique()
            
            probs = net(data_prime.x, data_prime.edge_index, data_prime.batch, None, penalty_coeff)
            retdict = criterion(probs, data_prime.edge_index, data_prime.batch, penalty_coeff, device)
            sets, set_edges, set_cardinality = decode_clique_final_speed(data_prime,(retdict["output"][0]), weight_factor =0.,draw=False)
            if set_cardinality.item() > model_output[model_index]:
                model_output[model_index] = set_cardinality
        end_t = time.time()
        time_per_data = time_per_data + end_t - start_t
        
        time_list.append(time_per_data)
        
        print('model_index:'+str(model_index)+' model:'+str(model_output[model_index])+"time:"+str(time_per_data))
        
    path_to_list = './dataset/testset/'
    if not os.path.exists(path_to_list+'testlist.pkl'):
        test_data_clique = []
        for data in testdata:
            my_graph = to_networkx(Data(x=data.x, edge_index = data.edge_index)).to_undirected()
            cliqno, _ = solve_gurobi_maxclique(my_graph, 500)
            data.clique_number = cliqno
            test_data_clique += [data]
        stored_dataset = open(path_to_list + 'testlist.pkl', 'wb')   
        dataset = pickle.dump(test_data_clique, stored_dataset)
    else:
        stored_dataset = open(path_to_list + 'testlist.pkl', 'rb')   
        test_data_clique = pickle.load(stored_dataset)

    ratios = [model_output[i]/test_data_clique[i].clique_number for i in range(len(model_output))]
    print(f"Mean ratio: {(np.array(ratios)).mean()} +/-  {(np.array(ratios)).std()}")

    print('avg_time:')
    print(np.mean(time_list))

if __name__ == '__main__':
    main()