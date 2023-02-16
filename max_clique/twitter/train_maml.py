import numpy as np
import torch
import random
import pickle
import argparse
import pathlib

from tqdm import tqdm
import sys
from dataset.twitter_train import TWITTER
from dataset.twitter_val import TWITTER_val
from learner_model import Learner
from gnn_model import clique_MPNN, ErdosLoss_clique
from utils import DataLog
import utils
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from torch_geometric.data import DataListLoader, DataLoader, Data

from utils import get_diracs, decode_clique_final_speed, solve_gurobi_maxclique
from torch.nn.utils.convert_parameters import (vector_to_parameters, parameters_to_vector)
from torch_geometric.utils import to_networkx

def main():
    
    logger = DataLog()

    # ===================
    # hyperparameters
    # ===================
    parser = argparse.ArgumentParser(description='Implicit MAML on Omniglot dataset')
    parser.add_argument('--data_dir', type=str, default='/home/aravind/data/omniglot-py/',
                        help='location of the dataset')
    parser.add_argument('--N_way', type=int, default=5, help='number of classes for few shot learning tasks')
    parser.add_argument('--K_shot', type=int, default=1, help='number of instances for few shot learning tasks')
    parser.add_argument('--inner_lr', type=float, default=1e-2, help='inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=1e-2, help='outer loop learning rate')
    parser.add_argument('--n_steps', type=int, default=16, help='number of steps in inner loop')
    parser.add_argument('--meta_steps', type=int, default=1000, help='number of meta steps')
    parser.add_argument('--task_mb_size', type=int, default=16)
    parser.add_argument('--lam', type=float, default=1.0, help='regularization in inner steps')
    parser.add_argument('--cg_steps', type=int, default=5)
    parser.add_argument('--cg_damping', type=float, default=1.0)
    parser.add_argument('--use_gpu', type=int, default=0)
    parser.add_argument('--num_tasks', type=int, default=20000)
    parser.add_argument('--save_dir', type=str, default='/tmp')
    parser.add_argument('--lam_lr', type=float, default=0.0)
    parser.add_argument('--lam_min', type=float, default=0.0)
    parser.add_argument('--scalar_lam', type=bool, default=True, help='keep regularization as a scalar or diagonal matrix (vector)')
    parser.add_argument('--taylor_approx', type=bool, default=False, help='Use Neumann approximation for (I+eps*A)^-1')
    parser.add_argument('--inner_alg', type=str, default='gradient', help='gradient or sqp for inner solve')
    parser.add_argument('--load_agent', type=str, default=None)
    parser.add_argument('--load_tasks', type=str, default=None)
    parser.add_argument('--seed', type=str, default=None)
    args = parser.parse_args()
    logger.log_exp_args(args)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))

    print("Generating tasks ...... ")
    cfg = Path('./dataset/configs/config.yaml')
    cfg_dict = yaml.safe_load(cfg.open('r'))
    
    dataset = TWITTER(cfg_dict['train'])
    testset = TWITTER_val(cfg_dict['val'])
    test_loader = DataLoader(testset, 1, shuffle=False)

    numlayers = 4
    penalty_coeff = 4.
    hidden_1 = 64
    hidden_2 = 1
    receptive_field = numlayers + 1


    if args.load_agent is None:
        learner_net = clique_MPNN(dataset, numlayers, hidden_1, hidden_2, 1)
        fast_net = clique_MPNN(dataset, numlayers, hidden_1, hidden_2, 1)
        meta_learner = Learner(model=learner_net, loss_function=torch.nn.CrossEntropyLoss(), inner_alg=args.inner_alg,
                            inner_lr=args.inner_lr, outer_lr=args.outer_lr, GPU=args.use_gpu)
        fast_learner = Learner(model=fast_net, loss_function=torch.nn.CrossEntropyLoss(), inner_alg=args.inner_alg,
                            inner_lr=args.inner_lr, outer_lr=args.outer_lr, GPU=args.use_gpu)
    else:
        meta_learner = pickle.load(open(args.load_agent, 'rb'))
        meta_learner.set_params(meta_learner.get_params())
        fast_learner = pickle.load(open(args.load_agent, 'rb'))
        fast_learner.set_params(fast_learner.get_params())
        for learner in [meta_learner, fast_learner]:
            learner.inner_alg = args.inner_alg
            learner.inner_lr = args.inner_lr
            learner.outer_lr = args.outer_lr
        
    init_params = meta_learner.get_params()
    #device = 'cuda' if args.use_gpu is True else 'cpu'
    device = torch.device('cuda:'+str(args.use_gpu) if torch.cuda.is_available() else 'cpu')
    lam = torch.tensor(args.lam) if args.scalar_lam is True else torch.ones(init_params.shape[0])*args.lam
    lam = lam.to(device)
    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ===================
    # Train
    # ===================
    print("Training model ......")
    losses = np.zeros((args.meta_steps, 2))
    num_tasks = len(dataset)
    
    highest_score = 0
    lowest_loss = 10000


    test_data_clique = []
    for data in test_loader:
        my_graph = to_networkx(Data(x=data.x, edge_index = data.edge_index)).to_undirected()
        cliqno, _ = solve_gurobi_maxclique(my_graph, 500)
        data.clique_number = cliqno
        test_data_clique += [data]

    for outstep in tqdm(range(args.meta_steps)):
        if outstep > 10 and outstep % 100 == 0:
            penalty_coeff = penalty_coeff + 0.0
        w1_list = []
        d1_list = []
        w2_list = []
        d2_list = []
        task_mb = np.random.choice(num_tasks, size=args.task_mb_size)
        w_k = meta_learner.get_params()
        meta_grad = 0.0
        lam_grad = 0.0
        for k in range(args.n_steps):
            old_parameters = parameters_to_vector(meta_learner.model.parameters())
            losses_q = torch.tensor([0.0]).to(device)
            data_index = 0
            for idx in task_mb:
                task = dataset[idx] # get task
                task = task.to(device)
                tl_before, w1, d1 = meta_learner.get_loss(task['train_x'], task['train_edge_index'], task['train_batch'], penalty_coefficient = penalty_coeff, device = device)
                new_grad = torch.autograd.grad(tl_before, meta_learner.model.parameters(), retain_graph = True, create_graph = True)
                new_params = parameters_to_vector(meta_learner.model.parameters()) - args.inner_lr * parameters_to_vector(new_grad)
                vector_to_parameters(new_params, meta_learner.model.parameters())
                tl_after, w2, d2 = meta_learner.get_loss(task['train_x'], task['train_edge_index'], task['train_batch'], penalty_coefficient = penalty_coeff, device = device)
                tl_after = tl_after.reshape(-1,1)
                if k == 0:
                    losses[outstep] += (np.array([tl_before.item(), 0])/args.task_mb_size)
                if k == args.n_steps - 1:
                    losses[outstep] += (np.array([0, tl_after.item()])/args.task_mb_size)
                if data_index == 0:
                    losses_q = tl_after
                else:
                    losses_q = torch.cat((losses_q, tl_after), 0)
                vector_to_parameters(old_parameters, meta_learner.model.parameters())
                data_index = data_index + 1

                w1_list.append(w1)
                w2_list.append(w2)
                d1_list.append(d1)
                d2_list.append(d2)
            loss_q = torch.mean(losses_q)
            meta_learner.outer_opt.zero_grad()
            loss_q.backward()
            meta_learner.outer_opt.step()
            print('loss:'+str(loss_q.item())+' w1:'+str(np.mean(w1_list))+' d1:'+str(np.mean(d1_list))+' w2:'+str(np.mean(w2_list))+' d2:'+str(np.mean(d2_list)))
            if loss_q.item() < lowest_loss:
                lowest_loss = loss_q.item()
                model_path = args.save_dir + '/best_loss.pth'
                torch.save(meta_learner.model.state_dict(), model_path)

        logger.log_kv('train_pre', losses[outstep,0])
        logger.log_kv('train_post', losses[outstep,1])
        
        if (outstep % 25 == 0 and outstep > 0) or outstep == args.meta_steps-1:
            smoothed_losses = utils.smooth_vector(losses[:outstep], window_size=10)
            plt.figure(figsize=(10,6))
            plt.plot(smoothed_losses)
            plt.ylim([-70, 300])
            plt.xlim([0, args.meta_steps])
            plt.grid(True)
            plt.legend(['Train pre', 'Train post'], loc=1)
            plt.savefig(args.save_dir+'/learn_curve.png', dpi=100)
            plt.clf()
            plt.close('all')

            pickle.dump(meta_learner, open(args.save_dir+'/agent.pickle', 'wb'))
            logger.save_log()

        if (outstep % 50 == 0):
            model_path = args.save_dir + '/model_' + str(outstep) + '.pth'
            torch.save(meta_learner.model.state_dict(), model_path)
        if outstep == args.meta_steps-1:
            model_path = args.save_dir + '/final_model.pth'
            torch.save(meta_learner.model.state_dict(), model_path)
        
        if outstep%5==0:
            model_output = np.zeros(len(testset))
            gt_output = []
            model_index = -1
            time_list = []
            if outstep % 1 == 0:
                for data in test_loader:
                    model_index = model_index + 1
                    for k in range(1):
                        # get k different data input
                        data_prime = get_diracs(data.to(device), 1, sparse = True, effective_volume_range=0.15, receptive_field = receptive_field)
                        data_prime = data_prime.to(device)
                        criterion = ErdosLoss_clique()
                        probs = meta_learner.model(data_prime.x, data_prime.edge_index, data_prime.batch, None, penalty_coeff)
                        retdict = criterion(probs, data_prime.edge_index, data_prime.batch, penalty_coeff, device)
                        sets, set_edges, set_cardinality = decode_clique_final_speed(data_prime,(retdict["output"][0]), weight_factor =0.,draw=False, beam = 1)
                        if set_cardinality.item() > model_output[model_index]:
                            model_output[model_index] = set_cardinality
                tests = test_data_clique
                ratios = [model_output[i]/tests[i].clique_number for i in range(len(tests))]
                print(f"Mean ratio: {(np.array(ratios)).mean()} +/-  {(np.array(ratios)).std()}")
                if (np.array(ratios)).mean() > highest_score:
                    highest_score = (np.array(ratios)).mean()
                    model_path = args.save_dir + '/best_model'+str(outstep)+'.pth'
                    torch.save(meta_learner.model.state_dict(), model_path)
                    print("epoch:"+str(outstep)+", get best again")

if __name__ == '__main__':
    main()