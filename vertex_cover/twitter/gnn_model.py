import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch import tensor
from torch.optim import Adam
from torch.optim import SGD
from math import ceil
from torch.nn import Linear
from torch.distributions import categorical
from torch.distributions import Bernoulli
import torch.nn
from torch_geometric.utils import convert as cnv
from torch_geometric.utils import sparse as sp
from torch_geometric.data import Data
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, GATConv
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.data import Batch 
from torch_scatter import scatter_min, scatter_max, scatter_add, scatter_mean
from torch import autograd
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops, segregate_self_loops, remove_isolated_nodes, contains_isolated_nodes, add_remaining_self_loops, dropout_adj
from utils import get_mask
from torch_geometric.nn.norm.graph_size_norm import GraphSizeNorm


class ErdosLoss_vertex(torch.nn.Module):
    def __init__(self):
        super(ErdosLoss_vertex,self).__init__()
        #self.penalty = Penalty()
    def forward(self, probs, edge_index, batch, penalty_coefficient, device):
        #calculating the terms for the vertex covering problem
        num_graphs = batch.max().item() + 1
        no_loop_index,_ = remove_self_loops(edge_index)  
        no_loop_row, no_loop_col = no_loop_index
        probs_sum = torch.zeros(num_graphs, device = device)
        #pairwise_prodsums = torch.zeros(num_graphs, device = device)
        for graph in range(num_graphs):
            batch_graph = (batch==graph)
            probs_sum[graph] = probs[batch_graph].unsqueeze(-1).sum()
        vertex_row = probs[no_loop_row]
        vertex_col = probs[no_loop_col]
        expected_distance = (1 - vertex_row) * (1 - vertex_col)
        expected_distance = expected_distance.sum() / num_graphs
        expected_weight = probs_sum.mean()
        loss = penalty_coefficient * expected_distance + expected_weight
        retdict = {}
        retdict["loss"] = [loss.squeeze(),"sequence"] #final loss
        retdict["Expected weight"]= [expected_weight, "sequence"]
        retdict["Expected distance"]= [expected_distance, "sequence"]
        return retdict

class ErdosLoss_vertex_new(torch.nn.Module):
    def __init__(self):
        super(ErdosLoss_vertex_new,self).__init__()
        #self.penalty = Penalty()
    def forward(self, probs, edge_index, batch, penalty_coefficient, device):
        #calculating the terms for the vertex covering problem
        num_graphs = batch.max().item() + 1
        no_loop_index,_ = remove_self_loops(edge_index)  
        no_loop_row, no_loop_col = no_loop_index
        vertex_row = probs[no_loop_row]
        vertex_col = probs[no_loop_col]
        expected_distance = (1 - vertex_row) * (1 - vertex_col)
        expected_distance = expected_distance.sum() / num_graphs
        expected_weight = probs.sum() / num_graphs
        loss = penalty_coefficient * expected_distance + expected_weight
        return loss
        


class vertex_MPNN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden1, hidden2, deltas):
        super(vertex_MPNN, self).__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.momentum = 0.1
        self.convs = torch.nn.ModuleList()
        self.deltas = deltas
        self.numlayers = num_layers
        self.heads = 8
        self.concat = True
        
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers-1):
            self.bns.append(BN(self.heads*self.hidden1, momentum=self.momentum))
        self.convs = torch.nn.ModuleList()        
        for i in range(num_layers - 1):
                self.convs.append(GINConv(Sequential(
            Linear( self.heads*self.hidden1,  self.heads*self.hidden1),
            ReLU(),
            Linear( self.heads*self.hidden1,  self.heads*self.hidden1),
            ReLU(),
            BN(self.heads*self.hidden1, momentum=self.momentum),
        ),train_eps=True))
        self.bn1 = BN(self.heads*self.hidden1)       
        self.conv1 = GINConv(Sequential(Linear(self.hidden2,  self.heads*self.hidden1),
            ReLU(),
            Linear( self.heads*self.hidden1,  self.heads*self.hidden1),
            ReLU(),
            BN(self.heads*self.hidden1, momentum=self.momentum),
        ),train_eps=True)

        if self.concat:
            self.lin1 = Linear(self.heads*self.hidden1, self.hidden1)
        else:
            self.lin1 = Linear(self.hidden1, self.hidden1)
        self.lin2 = Linear(self.hidden1, 1)
        self.gnorm = GraphSizeNorm()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        
        for conv in self.convs:
            conv.reset_parameters() 
        for bn in self.bns:
            bn.reset_parameters()
        self.bn1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch, edge_dropout = None, penalty_coefficient = 0.25):
        num_graphs = batch.max().item() + 1
        row, col = edge_index     
        total_num_edges = edge_index.shape[1]
        N_size = x.shape[0]

        if edge_dropout is not None:
            edge_index = dropout_adj(edge_index, edge_attr = (torch.ones(edge_index.shape[1], device=device)).long(), p = edge_dropout, force_undirected=True)[0]
            edge_index = add_remaining_self_loops(edge_index, num_nodes = batch.shape[0])[0]
                
        reduced_num_edges = edge_index.shape[1]
        current_edge_percentage = (reduced_num_edges/total_num_edges)
        no_loop_index,_ = remove_self_loops(edge_index)  
        no_loop_row, no_loop_col = no_loop_index

        xinit= x.clone()
        x = x.unsqueeze(-1)
        mask = get_mask(x,edge_index,1).to(x.dtype)
        x = F.leaky_relu(self.conv1(x, edge_index))# +x
        x = x*mask
        x = self.gnorm(x)
        x = self.bn1(x)
        
            
        for conv, bn in zip(self.convs, self.bns):
            if(x.dim()>1):
                x =  x+F.leaky_relu(conv(x, edge_index))
                mask = get_mask(mask,edge_index,1).to(x.dtype)
                x = x*mask
                x = self.gnorm(x)
                x = bn(x)

        xpostconvs = x.detach()
        #
        x = F.leaky_relu(self.lin1(x)) 
        x = x*mask

        xpostlin1 = x.detach()
        x = F.leaky_relu(self.lin2(x)) 
        x = x*mask

        #calculate min and max
        batch_max = scatter_max(x, batch, 0, dim_size= N_size)[0]
        batch_max = torch.index_select(batch_max, 0, batch)        
        batch_min = scatter_min(x, batch, 0, dim_size= N_size)[0]
        batch_min = torch.index_select(batch_min, 0, batch)

        #min-max normalize
        x = (x-batch_min)/(batch_max+1e-6-batch_min)
        probs=x

        return probs