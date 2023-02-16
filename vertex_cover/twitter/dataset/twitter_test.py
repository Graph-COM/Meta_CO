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

def optimize(vertices, edges):
    m = Model()
    vertexVars = {}
    for v in vertices:
        vertexVars[v] = m.addVar(vtype=GRB.BINARY,obj=1.0, name="x%d" % v)
    m.update()
    num_edge = edges.shape[0]
    for edge_index in range(num_edge):
        u = edges[edge_index][0]
        v = edges[edge_index][1]
        xu = vertexVars[u]
        xv = vertexVars[v]
        m.addConstr(xu + xv >= 1, name="e%d-%d" % (u, v))

    m.update()
    m.write('test.lp')
    m.optimize()

    cover = []
    for v in vertices:
        #print( vertexVars[v].X)
        if vertexVars[v].X > 0.5:
            print ("Vertex'," +str(v)+ 'is in the cover')
            cover.append(v)
    return cover

class TWITTER_test(InMemoryDataset):
    def __init__(self, config:dict):
        self.config = config
        self.data_path = Path(config['data_dir'])
        super(TWITTER_test, self).__init__(root=self.data_path)
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

        dataset_name = 'TWITTER_SNAP'
        path_to_dataset = './raw_dataset/'
        stored_dataset = open(path_to_dataset + 'TWITTER_SNAP.p', 'rb')   
        dataset = pickle.load(stored_dataset)

        total_sample = len(dataset)
        train_sample = int(0.8*total_sample)
        testset = dataset[train_sample:]
        test_length = len(testset)
        # for each task, it's a separate dataset
        data_list = []
        all_x = []
        all_cover = []
        for task_index in tqdm(range(test_length)):
            '''
            if task_index % 50 == 0:
                print(str(task_index)+'done')
            '''
            x = testset[task_index].x
            edge_index = testset[task_index].edge_index
            num_vertex = x.shape[0]
            edges = np.transpose(edge_index.numpy())
            vertex = range(num_vertex)
            cover = optimize(vertex, edges)
            min_cover = len(cover)
            #import pdb; pdb.set_trace()
            final_data = Data(x = x, edge_index = edge_index, min_cover = min_cover)
            data_list.append(final_data)
            all_x.append(x.shape[0])
            all_cover.append(min_cover)
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
            dataset = TWITTER_test(cfg_dict['test'])
