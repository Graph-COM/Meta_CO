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

### STEP 1: data Preparation

#### 1. 1 TWITTER

for twitter dataset, go to /max_clique/twitter/dataset/configs/config.yaml

enter the path to the folder that you would like to save the PyG form dataset in this yaml file, an example of the config.yaml file is shown as follows:
```
train:
    target_path: /...[your path].../meta_CO/max_clique/twitter/dataset/trainset
    data_dir: /...[your path].../meta_CO/max_clique/twitter/dataset/trainset

val:
    target_path: /...[your path].../meta_CO/max_clique/twitter/dataset/valset
    data_dir: /...[your path].../meta_CO/max_clique/twitter/dataset/valset

test:
    target_path: /...[your path].../meta_CO/max_clique/twitter/dataset/testset
    data_dir: /...[your path].../meta_CO/max_clique/twitter/dataset/testset
```

split and transform the original twitter dataset (which is shuffled and saved in /max_clique/twitter/dataset/raw_dataset, it needs unzip first), go to /max_clique/twitter/dataset and run:

```
python twitter_test.py
python twitter_train.py
python twitter_val.py
```

#### 1. 2 RB200/500
for RB dataset, the data generation is adopted from [the github repository of RUN-CSP](https://github.com/RUNCSP/RUN-CSP/blob/master/generate_xu_instances.py).

here we use RB200 as an example:
go to /max_clique/rb200/dataset/configs/config.yaml

enter the path to the folder that you would like to save the PyG form dataset in this yaml file, an example of the config.yaml file is shown as follows:
```
train:
    target_path: /...[your path].../meta_CO/max_clique/rb200/dataset/trainset
    data_dir: /...[your path].../meta_CO/max_clique/rb200/dataset/trainset

val:
    target_path: /...[your path].../meta_CO/max_clique/rb200/dataset/valset
    data_dir: /...[your path].../meta_CO/max_clique/rb200/dataset/valset

test:
    target_path: /...[your path].../meta_CO/max_clique/rb200/dataset/testset
    data_dir: /...[your path].../meta_CO/max_clique/rb200/dataset/testset
```

go to /max_clique/rb200/dataset to generate the training,val / testing data:

```
sh rb200_test.sh
sh rb200_train.sh
sh rb200_val.sh
```

### STEP2: training
go to /max_clique/[dataset] folder, edit the maml.sh doc as you may want to, fill in the GPU number in utils.py (it should align with the gpu number in maml.sh) you are going to use, then run
```
sh maml.sh
```

To tun the [Erdos goes neural (EGN)](https://github.com/Stalence/erdos_neu) baseline, run
```
sh erdos.sh
```
### STEP3: testing
go to /max_clique/[dataset] folder, open test.py, fill in the path to the model that you would like to test, then edit test.sh file, run
```
sh test.sh
```
for the problems, we provide our pre-trained model, in /max_clique/[dataset]/train_files/maml(erdos)/demo/best_model.pth, which could be used to directly solve the problems.

### STEP4: fine-tuning
go to /max_clique/[dataset] folder, open finetune.py, fill in the path to the model that you would like to fine-tune, then edit finetune.sh file, run
```
sh finetune.sh
```


## The Minimum Vertex Covering (MVC)

### STEP 1: data Preparation

#### 1. 1 TWITTER
for twitter dataset, go to /vertex_cover/twitter/dataset/configs/config.yaml

enter the path to the folder that you would like to save the PyG form dataset in this yaml file, an example of the config.yaml file is shown as follows:
```
train:
    target_path: /...[your path].../meta_CO/vertex_cover/twitter/dataset/trainset
    data_dir: /...[your path].../meta_CO/vertex_cover/twitter/dataset/trainset

val:
    target_path: /...[your path].../meta_CO/vertex_cover/twitter/dataset/valset
    data_dir: /...[your path].../meta_CO/vertex_cover/twitter/dataset/valset

test:
    target_path: /...[your path].../meta_CO/vertex_cover/twitter/dataset/testset
    data_dir: /...[your path].../meta_CO/vertex_cover/twitter/dataset/testset
```

split and transform the original twitter dataset (which is shuffled and saved in /vertex_cover/twitter/dataset/raw_dataset, it needs unzip first), go to /vertex_cover/twitter/dataset and run:

```
python twitter_test.py
python twitter_train.py
python twitter_val.py
```

#### 1. 2 RB200/500
for RB dataset, the data generation is adopted from [the github repository of RUN-CSP](https://github.com/RUNCSP/RUN-CSP/blob/master/generate_xu_instances.py).

here we use RB200 as an example:
go to /vertex_cover/rb200/dataset/configs/config.yaml

enter the path to the folder that you would like to save the PyG form dataset in this yaml file, an example of the config.yaml file is shown as follows:
```
train:
    target_path: /...[your path].../meta_CO/vertex_cover/rb200/dataset/trainset
    data_dir: /...[your path].../meta_CO/vertex_cover/rb200/dataset/trainset

val:
    target_path: /...[your path].../meta_CO/vertex_cover/rb200/dataset/valset
    data_dir: /...[your path].../meta_CO/vertex_cover/rb200/dataset/valset

test:
    target_path: /...[your path].../meta_CO/vertex_cover/rb200/dataset/testset
    data_dir: /...[your path].../meta_CO/vertex_cover/rb200/dataset/testset
```

go to /vertex_cover/rb200/dataset to generate the training,val / testing data:

```
sh rb200_test.sh
sh rb200_train.sh
sh rb200_val.sh
```

### STEP2: training
go to /vertex_cover/[dataset] folder, edit the maml.sh doc as you may want to, fill in the GPU number in utils.py (it should align with the gpu number in maml.sh) you are going to use, then run
```
sh maml.sh
```

To tun the [Erdos goes neural (EGN)](https://github.com/Stalence/erdos_neu) baseline, run
```
sh erdos.sh
```
### STEP3: testing
go to /vertex_cover/[dataset] folder, open test.py, fill in the path to the model that you would like to test, then edit test.sh file, run
```
sh test.sh
```
for the problems, we provide our pre-trained model, in /vertex_cover/[dataset]/train_files/erdos(maml)/demo/best_model.pth, which could be used to directly solve the problems.

### STEP4: fine-tuning
go to /vertex_cover/[dataset] folder, open finetune.py, fill in the path to the model that you would like to fine-tune, then edit finetune.sh file, run
```
sh finetune.sh
```


