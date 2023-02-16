# generate application 1 dataset
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import pickle
from pathlib import Path
import yaml
import re
import itertools
from torch_geometric.data import DataLoader
from utils import get_diracs
from tqdm import tqdm
from gurobipy import * 
import gurobipy as gp
from gurobipy import GRB
from torch_geometric.datasets import TUDataset
import pulp
import networkx as nx
import random

def generate_instance(n, k, r, p):
    '''
    n: number of cliques
    k: number of nodes in each clique
    a: log(k)/log(n)
    s: in each sampling iteration, the number of edges to be added
    iterations: how many iteration to sample
    return: the single-directed edges in numpy array form
    '''
    a = np.log(k) / np.log(n)
    v = k * n
    s = int(p * (n ** (2 * a)))
    iterations = int(r * n * np.log(n) - 1)
    parts = np.reshape(np.int64(range(v)), (n, k))
    nand_clauses = []
    
    for i in parts:
        nand_clauses += itertools.combinations(i, 2)
    edges = set()
    for _ in range(iterations):
        i, j = np.random.choice(n, 2, replace=False)
        all = set(itertools.product(parts[i, :], parts[j, :]))
        all -= edges
        edges |= set(random.sample(tuple(all), k=min(s, len(all))))

    nand_clauses += list(edges)
    clauses = np.array(nand_clauses)
    return clauses


class RB200_train(InMemoryDataset):
    def __init__(self, config:dict):
        self.config = config
        self.data_path = Path(config['data_dir'])
        super(RB200_train, self).__init__(root=self.data_path)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['data.pt']
    def download(self):
        # Download to `self.raw_dir`.
        pass
    def get_idx_split(self, split_type = 'Random'):
        data_idx = np.arange(2389)
        train_idx = data_idx
        return {'train':torch.tensor(train_idx,dtype = torch.long)}
    def process(self):
        # for each task, it's a separate dataset
        data_list = []
        
        for task_index in tqdm(range(2000)):
            '''
            n = np.random.randint(2x, 2x + d)
            k = np.random.randint(x, x+d)
            '''
            #n = np.random.randint(10, 26)
            #k = np.random.randint(5, 21)
            n = np.random.randint(6, 16)
            k = np.random.randint(12, 22)
            p = np.random.uniform(0.3, 1.0)
            #p = 0.8
            a = np.log(k) / np.log(n)
            r = - a / np.log(1 - p)
            edges = generate_instance(n, k, r, p)
            vertex = range(n*k)
            edges = torch.tensor(edges.transpose())
            reversed_edges = edges[[1,0]]
            graph_edges = torch.cat((edges, reversed_edges), 1)
            x = torch.zeros(n*k).reshape(-1, 1)
            tmp_data_list = []
            tmp_data = Data(x = x, edge_index = graph_edges)
            tmp_data_list.append(tmp_data)
            tmp_data_loader = DataLoader(tmp_data_list, batch_size = 1)
            for data in tmp_data_loader:
                new_data = get_diracs(data, 1, sparse = True, effective_volume_range=0.15, receptive_field = 5)
                #import pdb; pdb.set_trace()
                final_data = Data(x = new_data.x, edge_index = new_data.edge_index, train_batch = new_data.batch)
                data_list.append(final_data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        #import pdb; pdb.set_trace()
# 85.24 / 130.21
if __name__ == '__main__':
    import os
    configs = Path('./configs')
    for cfg in configs.iterdir():
        if str(cfg).startswith("configs/config"):
            cfg_dict = yaml.safe_load(cfg.open('r'))
            dataset = RB200_train(cfg_dict['train'])
