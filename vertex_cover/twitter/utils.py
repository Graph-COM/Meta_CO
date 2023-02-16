import os
import numpy as np
import scipy
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
import csv
from torch_scatter import scatter_min, scatter_max, scatter_add, scatter_mean
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
import torch.nn.functional as F

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

def to_cuda(x):
    try:
        return x.cuda()
    except:
        return torch.from_numpy(x).float().cuda()


def to_tensor(x):
    if type(x) == np.ndarray:
        return torch.from_numpy(x).float()
    elif type(x) == torch.Tensor:
        return x
    else:
        print("Type error. Input should be either numpy array or torch tensor")
    

def to_device(x, GPU=-1):
    if GPU>=0:
        device = torch.device('cuda:'+str(GPU)if torch.cuda.is_available() else "cpu")
        return x.to(device)
    else:
        return to_tensor(x)
    
    
def to_numpy(x):
    if type(x) == np.ndarray:
        return x
    else:
        try:
            return x.data.numpy()
        except:
            return x.cpu().data.numpy()

def cg_solve(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10, x_init=None):
    """
    Goal: Solve Ax=b equivalent to minimizing f(x) = 1/2 x^T A x - x^T b
    Assumption: A is PSD, no damping term is used here (must be damped externally in f_Ax)
    Algorithm template from wikipedia
    Verbose mode works only with numpy
    """
       
    if type(b) == torch.Tensor:
        x = torch.zeros(b.shape[0]) if x_init is None else x_init
        x = x.to(b.device)
        if b.dtype == torch.float16:
            x = x.half()
        r = b - f_Ax(x)
        p = r.clone()
    elif type(b) == np.ndarray:
        x = np.zeros_like(b) if x_init is None else x_init
        r = b - f_Ax(x)
        p = r.copy()
    else:
        print("Type error in cg")

    fmtstr = "%10i %10.3g %10.3g %10.3g"
    titlestr = "%10s %10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm", "obj fn"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
            norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
            print(fmtstr % (i, r.dot(r), norm_x, obj_fn))

        rdotr = r.dot(r)
        Ap = f_Ax(p)
        alpha = rdotr/(p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        newrdotr = r.dot(r)
        beta = newrdotr/rdotr
        p = r + beta * p

        if newrdotr < residual_tol:
            # print("Early CG termination because the residual was small")
            break

    if callback is not None:
        callback(x)
    if verbose: 
        obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
        norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
        print(fmtstr % (i, r.dot(r), norm_x, obj_fn))
    return x


def smooth_vector(vec, window_size=25):
    svec = vec.copy()
    if vec.shape[0] < window_size:
        for i in range(vec.shape[0]):
            svec[i,:] = np.mean(vec[:i, :], axis=0)
    else:   
        for i in range(window_size, vec.shape[0]):
            svec[i,:] = np.mean(vec[i-window_size:i, :], axis=0)
    return svec

class DataLog:
    
    def __init__(self):
        self.log = {}
        self.max_len = 0
        
    def log_exp_args(self, parsed_args):
        args = vars(parsed_args) # makes it a dictionary
        for k in args.keys():
            self.log_kv(k, args[k])

    def log_kv(self, key, value):
        # logs the (key, value) pair
        if key not in self.log:
            self.log[key] = []
        self.log[key].append(value)
        if len(self.log[key]) > self.max_len:
            self.max_len = self.max_len + 1

    def save_log(self, save_path=None):
        save_path = self.log['save_dir'][-1] if save_path is None else save_path
        pickle.dump(self.log, open(save_path+'/log.pickle', 'wb'))
        with open(save_path+'/log.csv', 'w') as csv_file:
            fieldnames = self.log.keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in range(self.max_len):
                row_dict = {}
                for key in self.log.keys():
                    if row < len(self.log[key]):
                        row_dict[key] = self.log[key][row]
                writer.writerow(row_dict)

    def get_current_log(self):
        row_dict = {}
        for key in self.log.keys():
            row_dict[key] = self.log[key][-1]
        return row_dict

    def read_log(self, log_path):
        with open(log_path) as csv_file:
            reader = csv.DictReader(csv_file)
            listr = list(reader)
            keys = reader.fieldnames
            data = {}
            for key in keys:
                data[key] = []
            for row in listr:
                for key in keys:
                    try:
                        data[key].append(eval(row[key]))
                    except:
                        None
        self.log = data


def get_diracs(data, N , n_diracs = 1,  sparse = False, flat = False, replace = True, receptive_field = 7, effective_volume_range = 0.1, max_iterations=20, complement = False):
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

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
            
    #                 print("Gr: ", gr)
    #                 print("Graph edge index: ", graph_edge_index)
    #                 print("gr max: ", gr.max())



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
             
                
            
                #for batch prime
#                 dirac_indices = torch.arange(N, device=device).unsqueeze(-1).expand(-1, graph_nodes).contiguous().view(-1)
#                 dirac_indices = dirac_indices.long()
#                 dirac_indices += k*N
#                 batch_prime = torch.cat((batch_prime, dirac_indices))



            locationmatrix = diracmatrix.nonzero()
            if complement:
                return Batch(batch = batch_index, x = diracmatrix, edge_index = original_edge_index,
                             y = data.y, locations = locationmatrix, volume_range = volume_range, recfield_vol = recfield_vols, total_vol = total_vols, complement_edge_index = data.complement_edge_index)
            else:
                return Batch(batch = batch_index, x = diracmatrix, edge_index = original_edge_index,
                             y = data.y, locations = locationmatrix, volume_range = volume_range, recfield_vol = recfield_vols, total_vol = total_vols)
