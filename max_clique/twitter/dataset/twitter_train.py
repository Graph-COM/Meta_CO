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
class TWITTER(InMemoryDataset):
    def __init__(self, config:dict):
        self.config = config
        self.data_path = Path(config['data_dir'])
        super(TWITTER, self).__init__(root=self.data_path)
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
        train_sample = int(0.7*total_sample)
        trainset = dataset[:train_sample]
        # suppose we want 10000 tasks in total, each task has (5 train + 5valid = 10 samples), we need
        data_list = []
        for task_index in tqdm(range(train_sample)):
            train_list = []
            val_list = []
            for i in range(1):
                train_list.append(trainset[task_index])
            train_loader = DataLoader(train_list, batch_size = 1, shuffle = False)
            for data in train_loader:
                data_prime = get_diracs(data, 1, sparse = True, effective_volume_range=0.15, receptive_field = 5)
                train_x = data_prime.x
                train_batch = data_prime.batch
                train_edge_index = data_prime.edge_index
                train_locations = data_prime.locations
            final_data = Data(train_x = train_x, train_edge_index = train_edge_index, train_batch = train_batch, train_locations = train_locations,
                            x = train_x, edge_index = train_edge_index)
            data_list.append(final_data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        

if __name__ == '__main__':
    import os
    configs = Path('./configs')
    for cfg in configs.iterdir():
        if str(cfg).startswith("configs/config"):
            cfg_dict = yaml.safe_load(cfg.open('r'))
            dataset = TWITTER(cfg_dict['train'])
