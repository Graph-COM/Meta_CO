
import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.utils import sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import convert as cnv
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from torch.distributions import categorical
from torch.distributions import Bernoulli
from torch.distributions import relaxed_categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from matplotlib import pyplot as plt
import numpy as np
from torch_geometric.utils import is_undirected, to_undirected, softmax, add_self_loops, remove_self_loops, segregate_self_loops, remove_isolated_nodes, contains_isolated_nodes, add_remaining_self_loops
import time
import gurobipy as gp
from gurobipy import GRB

from torch import autograd
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.utils import dropout_adj
#from torch_geometric.utils import scatter
from torch_geometric.utils import degree
from torch_geometric.data import Batch 

from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing

from torch_scatter import scatter_min, scatter_max, scatter_add, scatter_mean

def propagate(x, edge_index):
    row, col = edge_index
    out = scatter_add(x[col], row, dim=0)
    return out

#get the k-hop neighbors of node sample
def get_mask(x, edge_index, hops):
    for k in range(hops):
        x = propagate(x, edge_index)
    mask = (x>0).float()
    return mask
def get_diracs(data, N , n_diracs = 1,  sparse = False, flat = False, replace = True, receptive_field = 7, effective_volume_range = 0.1, max_iterations=20, complement = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    if not sparse:
        # if not sparse
        graphcount =data.num_nodes #number of graphs in data/batch object
        totalnodecount = data.x.shape[1] #number of total nodes for each graph 
        actualnodecount = 0 #cumulative number of nodes
        diracmatrix= torch.zeros((graphcount,totalnodecount,N),device=device) #matrix with dirac pulses
        for k in range(graphcount):
            graph_nodes = data.mask[k].sum() #number of nodes in the graph
            actualnodecount += graph_nodes #might not need this, we'll see
            probabilities= torch.ones((graph_nodes.item(),1),device=device)/graph_nodes #uniform probs
            node_distribution=OneHotCategorical(probs=probabilities.squeeze())
            node_sample= node_distribution.sample(sample_shape=(N,))
            node_sample= torch.cat((node_sample,torch.zeros((N,totalnodecount-node_sample.shape[1]),device=device)),-1) #concat zeros to fit dataset shape
            diracmatrix[k,:]= torch.transpose(node_sample,dim0=-1,dim1=-2) #add everything to the final matrix           
        return diracmatrix
    
    else:
            if not is_undirected(data.edge_index):
                data.edge_index = to_undirected(data.edge_index)
              
            original_batch_index = data.batch
            original_edge_index = add_remaining_self_loops(data.edge_index, num_nodes = data.batch.shape[0])[0]
            #original_edge_index, _, node_mask = remove_isolated_nodes(original_edge_index)
            #batch_index = original_batch_index[node_mask]
            batch_index = original_batch_index
            
            graphcount = data.num_graphs #number of graphs in data/batch object
            batch_prime = torch.zeros(0,device=device).long()
            
            r,c = original_edge_index
            
            
            global_offset = 0
            all_nodecounts = scatter_add(torch.ones_like(batch_index,device=device), batch_index,dim=0)
            recfield_vols = torch.zeros(graphcount,device=device)
            total_vols = torch.zeros(graphcount,device=device)
            
            for j in range(n_diracs):
                diracmatrix = torch.zeros(0,device=device)
                locationmatrix = torch.zeros(0,device=device).long()
        
                for k in range(graphcount):
                    #get edges of current graph, remember to subtract offset
                    graph_nodes = all_nodecounts[k]
                    if graph_nodes==0:
                        print("all nodecounts: ", all_nodecounts)
                    graph_edges = (batch_index[r]==k)
                    graph_edge_index = original_edge_index[:,graph_edges]-global_offset           
                    gr, gc = graph_edge_index

                    #get dirac
                    randInt = np.random.choice(range(graph_nodes), N, replace = replace)
                    node_sample = torch.zeros(N*graph_nodes,device=device)
                    offs  = torch.arange(N, device=device)*graph_nodes
                    dirac_locations = (offs + torch.from_numpy(randInt).to(device))
                    node_sample[dirac_locations] = 1

                    #calculate receptive field volume and compare to total volume
                    mask = get_mask(node_sample, graph_edge_index.detach(), receptive_field).float()
                    deg_graph = degree(gr, (graph_nodes.item()))

                    total_volume = deg_graph.sum()
                    recfield_volume = (mask*deg_graph).sum()
                    volume_range = recfield_volume/total_volume
                    total_vols[k] = total_volume
                    recfield_vols[k] = recfield_volume

                    #if receptive field volume is less than x% of total volume, resample
                    for iteration in range(max_iterations):  
                        randInt = np.random.choice(range(graph_nodes), N, replace = replace)
                        node_sample = torch.zeros(N*graph_nodes,device=device)
                        offs  = torch.arange(N, device=device)*graph_nodes
                        dirac_locations = (offs + torch.from_numpy(randInt).to(device))
                        node_sample[dirac_locations] = 1

                        mask = get_mask(node_sample, graph_edge_index, receptive_field).float()
                        recfield_volume = (mask*deg_graph).sum()
                        volume_range = recfield_volume/total_volume
                        
                        if volume_range > effective_volume_range:
                            recfield_vols[k] = recfield_volume
                            total_vols[k] = total_volume
                            break;
                    dirac_locations2 = torch.from_numpy(randInt).to(device) + global_offset
                    global_offset += graph_nodes

                    diracmatrix = torch.cat((diracmatrix, node_sample),0)
                    locationmatrix = torch.cat((locationmatrix, dirac_locations2),0)
             
            locationmatrix = diracmatrix.nonzero()
            if complement:
                return Batch(batch = batch_index, x = diracmatrix, edge_index = original_edge_index,
                             y = data.y, locations = locationmatrix, volume_range = volume_range, recfield_vol = recfield_vols, total_vol = total_vols, complement_edge_index = data.complement_edge_index)
            else:
                return Batch(batch = batch_index, x = diracmatrix, edge_index = original_edge_index,
                             y = data.y, locations = locationmatrix, volume_range = volume_range, recfield_vol = recfield_vols, total_vol = total_vols)
