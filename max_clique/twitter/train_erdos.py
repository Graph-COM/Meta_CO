import torch
from torch.optim import Adam
from math import ceil
import torch.nn
import torch
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from dataset.twitter_train import TWITTER
from dataset.twitter_val import TWITTER_val
import scipy.io
import pickle
from torch_geometric.data import DataListLoader, DataLoader
from random import shuffle
from torch_geometric.datasets import TUDataset
import numpy as np
import yaml
from pathlib import Path
from gnn_model import clique_MPNN, ErdosLoss_clique
from utils import get_diracs, decode_clique_final_speed, solve_gurobi_maxclique
from torch_geometric.utils import dropout_adj, to_undirected, to_networkx
from torch_geometric.data import DataListLoader, DataLoader, Data
from tqdm import tqdm
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='this is the arg parser for max clique problem')
    parser.add_argument('--save_path', dest = 'save_path',default = 'train_files/erdos/try/')
    parser.add_argument('--gpu', dest = 'gpu',default = '0')
    parser.add_argument('--data_scale', dest = 'data_scale',default = 'all')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    # prepare the dataset
    cfg = Path('./dataset/configs/config.yaml')
    cfg_dict = yaml.safe_load(cfg.open('r'))
    if args.data_scale == 'all':
        dataset = TWITTER(cfg_dict['train'])
        testset = TWITTER_val(cfg_dict['val'])
    else:
        data_num = int(args.data_scale)
        dataset = TWITTER(cfg_dict['train'])
        dataset = dataset[:data_num]
        testset = TWITTER_val(cfg_dict['val'])
    train_loader = DataLoader(dataset, 128, shuffle = True)
    test_loader = DataLoader(testset, 1, shuffle=False)
    batch_size = 128
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')

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
    penalty_coeff = 4.
    hidden_1 = 64
    hidden_2 = 1

    net = clique_MPNN(dataset, numlayers, hidden_1, hidden_2, 1)
    net.to(device).reset_parameters()
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=0.00000)
    criterion = ErdosLoss_clique()

    test_data_clique = []
    for data in test_loader:
        my_graph = to_networkx(Data(x=data.x, edge_index = data.edge_index)).to_undirected()
        cliqno, _ = solve_gurobi_maxclique(my_graph, 500)
        data.clique_number = cliqno
        test_data_clique += [data]

    lowest_score = 0
    for epoch in tqdm(range(epochs)):
        totalretdict = {}
        #show currrent epoch and GPU utilizationss
        print('Epoch: ', epoch)
        #GPUtil.showUtilization()
        net.train()
        #train(net, train_loader, optimizer, epoch, device, receptive_field, penalty_coeff, totalretdict, batch_size, criterion)
        for data in train_loader:
            optimizer.zero_grad(), 
            data = data.to(device)
            data_prime = get_diracs(data, 1, sparse = True, effective_volume_range=0.15, receptive_field = receptive_field)
            data_prime = data_prime.to(device)
            probs = net(data_prime.x, data_prime.edge_index, data_prime.batch, None, penalty_coeff)
            #import pdb; pdb.set_trace()
            retdict = criterion(probs, data.edge_index, data.batch, penalty_coeff, device)
            data = data.to('cpu')
            for key,val in retdict.items():
                if "sequence" in val[1]:
                    if key in totalretdict:
                        totalretdict[key][0] += val[0].item()
                    else:
                        totalretdict[key] = [val[0].item(),val[1]]
            if epoch > 2:
                    retdict["loss"][0].backward()
                    #reporter.report()
                    torch.nn.utils.clip_grad_norm_(net.parameters(),1)
                    optimizer.step()
                    del(retdict)
        if epoch > -1:        
            for key,val in totalretdict.items():
                if "sequence" in val[1]:
                    val[0] = val[0]/(len(train_loader.dataset)/batch_size)
            del data_prime
        print(totalretdict.items())
        if epoch % 20==0:
            train_model_path = args.save_path + 'train_model'+str(epoch)+'.pth'
            torch.save(net.state_dict(), train_model_path)
        if epoch % 5 == 0:
            model_output = np.zeros(len(testset))
            gt_output = []
            model_index = -1
            time_list = []
            for data in test_loader:
                model_index = model_index + 1
                for k in range(8):
                    # get k different data input
                    data_prime = get_diracs(data.to(device), 1, sparse = True, effective_volume_range=0.15, receptive_field = receptive_field)
                    data_prime = data_prime.to(device)
                    criterion = ErdosLoss_clique()
                    probs = net(data_prime.x, data_prime.edge_index, data_prime.batch, None, penalty_coeff)
                    retdict = criterion(probs, data_prime.edge_index, data_prime.batch, penalty_coeff, device)
                    sets, set_edges, num_vertex = decode_clique_final_speed(data_prime,(retdict["output"][0]), weight_factor =0.,draw=False, beam = 1)
                    if num_vertex.item() > model_output[model_index]:
                        model_output[model_index] = num_vertex
                
            ratios = [(model_output[i])/test_data_clique[i].clique_number for i in range(len(model_output))]
            print(f"Mean ratio: {(np.array(ratios)).mean()} +/-  {(np.array(ratios)).std()}")
            if (np.array(ratios)).mean() > lowest_score:
                lowest_score = (np.array(ratios)).mean()
                model_path = args.save_path + 'best_model'+str(epoch)+'.pth'
                torch.save(net.state_dict(), model_path)
                print("epoch:"+str(epoch)+", get best again")
    train_model_path = args.save_path + 'train_model.pth'
    torch.save(net.state_dict(), train_model_path)


if __name__ == '__main__':
    main()