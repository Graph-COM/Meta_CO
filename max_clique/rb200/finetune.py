import torch
from torch.optim import Adam
import time
from math import ceil
import torch.nn
import torch
from utils import get_diracs
import scipy
import scipy.io
from torch_geometric.data import DataListLoader, DataLoader
from random import shuffle
import numpy as np
from gnn_model import clique_MPNN, ErdosLoss_clique
from utils import decode_clique_final_speed
import argparse
import os
from pathlib import Path
import yaml
from dataset.rb200_test import RB200_test


def main():
    parser = argparse.ArgumentParser(description='this is the arg parser for max clique problem')
    parser.add_argument('--save_path', dest = 'save_path',default = 'train_files/max_clique/new_train/')
    parser.add_argument('--gpu', dest = 'gpu',default = '0')
    args = parser.parse_args()

    
    # prepare the dataset
    cfg = Path('./dataset/configs/config.yaml')
    cfg_dict = yaml.safe_load(cfg.open('r'))
    testdata = RB200_test(cfg_dict['test'])
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
    learning_rate = 0.001
    numlayers = 4
    receptive_field = numlayers + 1
    penalty_coeff = 4.
    hidden_1 = 64
    hidden_2 = 1
    
    net = clique_MPNN(dataset, numlayers, hidden_1, hidden_2 ,1)
    state_dict = torch.load('./train_files/maml/demo/best_model.pth',map_location = torch.device('cpu'))
    net.train()
    #net.eval()
    model_output = np.zeros(len(testdata))
    #model_output = model_output + 10000
    gt_output = []
    model_index = -1
    time_list = []


    for data in test_loader:
        model_index = model_index + 1
        gt_output.append(data.max_clique.item())
        time_per_data = 0
        for k in range(8):
            # get k different data input
            data_prime = get_diracs(data.to(device), 1, sparse = True, effective_volume_range=0.15, receptive_field = receptive_field)
            data_prime = data_prime.to(device)
            net.reset_parameters()
            net.load_state_dict(state_dict)
            net.to(device)
            start_t = time.time()
            criterion = ErdosLoss_clique()
            optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=0.00000)
            probs = net(data_prime.x, data_prime.edge_index, data_prime.batch, None, penalty_coeff)
            retdict = criterion(probs, data_prime.edge_index, data_prime.batch, penalty_coeff, device)
            #num_vertex = decode_vertex_speed(data_prime,probs)
            sets, set_edges, num_vertex = decode_clique_final_speed(data_prime,(retdict["output"][0]), weight_factor =0.,draw=False)
            #num_vertex = decode_vertex(data_prime,probs)
            if num_vertex.item() > model_output[model_index]:
                model_output[model_index] = num_vertex
            for epoch in range(2):
                optimizer.zero_grad()
                probs = net(data_prime.x, data_prime.edge_index, data_prime.batch, None, penalty_coeff)
                retdict = criterion(probs, data_prime.edge_index, data_prime.batch, penalty_coeff, device)
                retdict["loss"][0].backward()
                optimizer.step()
                if epoch > 0:
                    probs = net(data_prime.x, data_prime.edge_index, data_prime.batch, None, penalty_coeff)
                    retdict = criterion(probs, data_prime.edge_index, data_prime.batch, penalty_coeff, device)
                    #num_vertex = decode_vertex(data_prime, probs)
                    sets, set_edges, num_vertex = decode_clique_final_speed(data_prime,(retdict["output"][0]), weight_factor =0.,draw=False)
                    if num_vertex.item() > model_output[model_index]:
                        model_output[model_index] = num_vertex
            end_t = time.time()
            time_per_data = time_per_data + end_t - start_t
            
        time_list.append(time_per_data)
        print('model_index:'+str(model_index)+' model:'+str(model_output[model_index])+" all_node:"+str(data.x.shape[0])+" time:"+str(time_per_data))
    #import pdb; pdb.set_trace()
    ratios_ = [model_output[i]/gt_output[i] for i in range(len(model_output))]
    ratios = [(model_output[i]) for i in range(len(model_output))]
    #ratios = ratios.numpy()
    print(f"Mean ratio: {(np.array(ratios_)).mean()} +/-  {(np.array(ratios_)).std()}")
    print(f"Mean ratio: {(np.array(ratios)).mean()} +/-  {(np.array(ratios)).std()}")
    
    
    print('avg_time:')
    print(np.mean(time_list))
    
if __name__ == '__main__':
    main()