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
        for task_index in range(test_length):
            if task_index % 50 == 0:
                print(str(task_index)+'done')
            x = testset[task_index].x
            edge_index = testset[task_index].edge_index
            #import pdb;pdb.set_trace()
            final_data = Data(x = x, edge_index = edge_index)
            data_list.append(final_data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        

if __name__ == '__main__':
    import os
    configs = Path('./configs')
    for cfg in configs.iterdir():
        if str(cfg).startswith("configs/config"):
            cfg_dict = yaml.safe_load(cfg.open('r'))
            dataset = TWITTER_test(cfg_dict['test'])
