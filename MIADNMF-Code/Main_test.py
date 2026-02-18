import os.path as osp
import torch
import argparse
import numpy as np 
import logging
import torch.nn.functional as F
from Getgraph.dataset import Dataset
from Model.Model import MyModel
from Metrics.evaluate import clusterscores
import matplotlib.pyplot as plt
from Pretrainer.Pretrainer import PreTrainer
from Model.Model import MyModel, Attention
from Model.function import k_means
from Param.Getparam import Getparam

def row_square_normalize(H, eps=1e-8):
    # 计算每行的平方和 [n,]
    row_norms = torch.sum(H ** 2, dim=1, keepdim=True)  
    # 归一化（除以平方和的根号）
    H_normalized = H / torch.sqrt(row_norms + eps)  
    return H_normalized

def train(model: MyModel, graph, optimizer):

    model.train()
    optimizer.zero_grad()
    h1, h2, h3, h_fuse, weights = model()
    # print(h1.shape, h2.shape, h3.shape, h_fuse.shape)
    loss = model.loss(h_fuse, weights)

    loss.backward()
    optimizer.step()

    # y_pred = np.argmax(h1.detach().cpu().numpy(), axis=0)
    
    h_fuse = row_square_normalize(h_fuse, eps=1e-6)
    y_pred = np.argmax(h_fuse.detach().cpu().numpy(), axis=0)
    y_true = graph.L.detach().cpu().numpy()
    scores = clusterscores(y_pred, y_true)

    return loss.item(), scores, weights


if __name__ == '__main__':

    need_pretrain = 0
    #data = 'washington'# 800 700//1000 700//1200 900//
    #data = 'wisconsin'#1000 600//1500 300
    #data = 'karate'
    #data = 'texas'#400 400//(400 1200 1)(400 400 0.001)
    data = 'cora'#800 1200//900 1000
    #data = 'cornell'#800 600//1100 900//1100 700(0.02)
    #data = 'football'#400 200
    #data = 'citeseer'#2000 1200(0.001)[512 64 6]//1600 700(0.001)[512 64 6]
    #data = 'gene'#1200 700
    #data = 'polbooks'#400 1300
    # data = 'terroristrel'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_config = {
        'graph_file': osp.join('./Database', data, 'edge.txt').replace('\\', '/'),
        'label_file': osp.join('./Database', data, 'group.txt').replace('\\', '/'),
        'device': device
    }
    graph = Dataset(dataset_config)

    model_config = {
        'A': graph.A,
        'device': device,
        'num_nodes': graph.num_nodes,
        'pretrain_params_path': osp.join('./Log', data, 'pretrain_params.pkl').replace('\\', '/'),
        'aloss': 1,
        'neg': 400,
        'tau': 1,
        'classes': graph.num_classes,
        'learning_rate': 0.01,
        'weight_decay': 1e-5,
        'epochs': 400,
        'times': 1,
        'is_init': True,
        'qloss': 10
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=data)
    parser.add_argument('--param', type=str, default='local:'+data+'.json')

    param_keys = model_config.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(model_config[key]), nargs='?')
    args = parser.parse_args()

    gp = Getparam(default=model_config)
    param = gp(source=args.param, preprocess='nni')

    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    pretrain_config = { 
        'pretrain_params_path': osp.join('./Log', data, 'pretrain_params.pkl').replace('\\', '/'),
        'A': graph.A,
        'device': device,
        'num_nodes': graph.num_nodes,
        'drop_rate': 0.1,
        'beta': 0.1,
        'layers': param['netshape'],
        'pre_iterations': 100,
        'seed': 42
    }

    logging.basicConfig(
        filename = 'run.log',
        level = logging.INFO,
        format = '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d',
        filemode='a'
    )
    if need_pretrain:
        pretrainer = PreTrainer(pretrain_config)
        pretrainer.pretrain('origin')
        pretrainer.pretrain('drop')
        pretrainer.pretrain('global')

    M = []
    N = []
    P = []
    F1 = []
    A = []
    finalloss = []

    learning_rate = param['learning_rate']
    weight_decay = param['weight_decay']

    for param['contra1'] in param['list']:
        for i in range(param['times']):

            # encoder = Encoder(param['num_nodes'], param['outdim'], 
            #                   param['activation'], 2)
            atten = Attention(param['classes'])
            model = MyModel(atten, param).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
                )   

            for epoch in range(param['epochs']):
                loss, scores, weights = train(model, graph, optimizer)
                finalloss.append(loss)
            
            M.append(scores['ACC'])
            N.append(scores['NMI'])
            P.append(scores['PUR'])
            F1.append(scores['F_score'])
            A.append(scores['ARI'])

    print("ACC:     ", np.round(M,4))
    print("PUR:     ", np.round(P,4))
    print("NMI:     ", np.round(N,4))
    print("F-score: ", np.round(F1,4))
    print("maxPUR:",max(P))

    # logging.info(f" dataset = {data} contra1 = {param['contra1']} epochs = {param['epochs']} layers = {param['netshape']}")
    # logging.info(f"ACC : {np.mean(M)} +- {np.var(M)}")
    # logging.info(f"PUR : {np.mean(P)} +- {np.var(P)}")
    # logging.info(f"F-score: {np.mean(F1)} +- {np.var(F1)}")
    # logging.info(f"weights: {np.round(torch.mean(weights, dim=0).tolist(), 4)} \n")

    # print("F-score : %f±%f" %(np.mean(F1) , np.var(F1)))
    # print("acc : %f±%f" %(np.mean(M) , np.var(M)))
    # print("NMI : %f±%f" %(np.mean(N) , np.var(N)))
    # print("PUR : %f±%f" %(np.mean(P) , np.var(P)))    
    #plt.plot(range(1,len(finalloss)+1), finalloss)
    #plt.show()
