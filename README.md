# Meta_CO
The official repository of the paper `unsupervised learning for combinatorial optimization needs meta learning' Haoyu Wang, Pan Li.

The repository contains the code and datasets contained in the max clique (MC), the minimum vertex covering (MVC), the max independent set (MIS) problems.

## Environments
The environment requires Pytorch, Pytorch Geometric and some of the following key packages:
```
torch                   1.9.0
torch-cluster           1.5.9
torch-geometric         1.7.2
torch-scatter           2.0.8
torch-sparse            0.6.11
torch-spline-conv       1.2.1
tqdm                    4.62.2
networkx                2.5.1
numpy                   1.20.3
ogb                     1.3.5
pandas                  1.3.0
scikit-learn            0.24.2
scipy                   1.6.3
PyYAML                  5.4.1
```
To run the Gurobi9.5 baseline, you may also need to install Gurobi package and require the Gurobi permit
```
gurobipy                9.5.1
```

## The Max Clique (MC)
