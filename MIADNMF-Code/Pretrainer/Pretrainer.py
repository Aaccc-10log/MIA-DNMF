from sklearn.decomposition import NMF
from tqdm import tqdm
import pickle
import networkx as nx
import torch
import numpy as np

class PreTrainer(object):
    def __init__(self, config):

        self.config = config
        self.A = config['A']
        #self.L = config['L']
        self.seed = config['seed']
        self.beta = config['beta']
        self.layers = config['layers']
        self.device = config['device']
        self.drop_rate = config['drop_rate']
        self.num_nodes = config['num_nodes']
        self.pre_iterations = config['pre_iterations']
        self.pretrain_params_path = config['pretrain_params_path']

        self.node_layer = torch.zeros(self.num_nodes, 1, device=self.device)
        self.edgeweight = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)
        self.globalsimilariy = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)
        
        self.node_layer = self.Onionshell()
        self.edgeweight=self.edge_weight(self.node_layer)
        self.adjdrop = self.drop_edge(self.edgeweight, self.drop_rate)
        # if self.num_nodes < 1000:
        #     self.k = self.get_k(K_min=5, K_max=10, lambda_factor=10.0)
        # elif self.num_nodes < 10000:
        #     self.k = self.get_k(K_min=5, K_max=50, lambda_factor=10.0)

        self.globalsimilariy = self.get_global(self.global_similariy(self.beta), self.get_k(self.A)+1).fill_diagonal_(0)

        self.U_init = {}
        self.V_init = {}

    # def get_k(self, K_min, K_max, lambda_factor=10.0):
    #     """
    #     输入：
    #         A: torch.Tensor，邻接矩阵 (N x N)，要求二值对称，无自环
    #         K_min, K_max: 最小、最大 k 值限制
    #         lambda_factor: 放缩系数 λ，用于调节 LDI 转换到 k 的范围

    #     输出：
    #         k_list: LongTensor (N,) 每个节点的 adaptive k
    #     """
    #     A = self.A
    #     N = A.shape[0]
    #     # 确保没有自环
    #     A = A.clone()
    #     A.fill_diagonal_(0)

    #     # 初始化 LDI 向量
    #     ldi_list = torch.zeros(N)

    #     for i in range(N):
    #         neighbors = (A[i] > 0).nonzero(as_tuple=True)[0]  # 邻居索引
    #         k_v = len(neighbors)

    #         if k_v < 2:
    #             ldi_list[i] = 0.0
    #             continue

    #         # 获取邻居诱导子图
    #         sub_A = A[neighbors][:, neighbors]  # (k_v x k_v)
    #         e_v = sub_A.sum() / 2  # 无向图边数

    #         clustering_factor = (2 * e_v) / (k_v * (k_v - 1))  # 聚类系数
    #         ldi = clustering_factor * torch.log1p(torch.tensor(float(k_v)))  # LDI 值

    #         ldi_list[i] = ldi

    #     # 将 LDI 映射到 k
    #     k_float = lambda_factor * ldi_list
    #     k_clipped = torch.clamp(k_float, min=K_min, max=K_max)
    #     k_final = k_clipped.floor().long()

    #     return k_final  # shape: (N,)
    # def get_global(self, S, k_list):
    #     N = S.shape[0]
    #     S = S.clone()
    #     mask = torch.zeros_like(S, dtype=torch.bool)
    #     for i in range(N):
    #         k = k_list[i].item()

    #         # 找出前k个最大相似度索引（注意不包括自己）
    #         topk = torch.topk(S[i], k=k, largest=True)
    #         idx = topk.indices
    #         mask[i, idx] = True
    #     # 应用掩码，只保留Top-k
    #     S_topk = torch.where(mask, S, torch.zeros_like(S))
    #     # 小于0的置零
    #     S_topk = torch.clamp(S_topk, min=0)
    #     return S_topk
    def setup_z(self, i, module):#第一层的输入是邻接矩阵，后续的输入是上一层的输出

        if module == 'origin':
            if i == 0:
                self.Z = self.A.detach().cpu().numpy()
            else:
                self.Z = self.V_init[module + str(i-1)]
        elif module == 'drop':
            if i == 0:
                self.Z = self.adjdrop.detach().cpu().numpy()
            else:
                self.Z = self.V_init[module + str(i-1)]
        elif module =='global':
            if i == 0:
                self.Z = self.globalsimilariy.detach().cpu().numpy()
            else:
                self.Z = self.V_init[module + str(i-1)]
            
    def sklearn_pretrain(self, i):
            """
            Pretraining a single layer of the model with sklearn.
            :param i: Layer index.
            """
            nmf_model = NMF(n_components=self.layers[i],
                            init="random",
                            random_state=self.seed,
                            max_iter=self.pre_iterations)
            
            U = nmf_model.fit_transform(self.Z)
            V = nmf_model.components_
            return U, V
    
    def Onionshell(self):

        A = self.A
        n = A.size(0)
        L = torch.zeros(n, dtype=torch.int)
        B = A.clone()
        d = 1
        l = 1
        
        while B.sum() != 0:
            C = B.sum(dim=1)
            
            # Find indices where C <= d and C != 0
            i = torch.where((C <= d) & (C != 0))[0]
            
            if i.numel() > 0:
                B[i, :] = 0
                B[:, i] = 0
                L[i] = l
                l += 1
                d = 1
            else:
                d += 1
        return L
    
    def edge_weight(self, node_layer):

        row, col = torch.where(self.A != 0)
        edge_weight = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)

        for i in range(len(row)):
            edge_weight[row[i],col[i]] = (node_layer[row[i]] + node_layer[col[i]])/2
        
        return edge_weight
    
    def drop_edge(self, edgeweight, drop_rate):
        values = edgeweight[edgeweight > 0]
        sort_values = torch.sort(values.flatten(), descending=True).values
        k = int((1-drop_rate) * len(sort_values))-1
        mask = torch.where(edgeweight > sort_values[k], torch.ones_like(edgeweight), 
                           torch.zeros_like(edgeweight)).to(self.device)
        A = self.A * mask
        return A
    
    def global_similariy(self, beta):
        A = self.A
        I = torch.eye(self.num_nodes, device=self.device)
        S = torch.linalg.inv(I - beta * A) - I
        return S
    
    #################################改
    def get_k(self, A):
        A = A.cpu().numpy()
        numnodes = A.shape[0]
        degree = np.sum(A, axis=1)
        avg = np.mean(degree)

        if numnodes < 1000:
            k = int(min(10, 2*avg))
        else:
            k = int(min(50, 20*avg))###########################################################

        return k

    def get_global(self, S, k):

        n = S.size(0)
        # 获取每行的 topk 值和索引
        S = torch.clamp(S, min=0)
        topk_values, topk_indices = torch.topk(S, k=k, dim=1)  # (n, k)
        
        # 创建一个全零矩阵，并填充 topk 值
        S_sparse = torch.zeros_like(S)
        row_indices = torch.arange(n).unsqueeze(1).expand(-1, k)  # (n, k)
        S_sparse[row_indices, topk_indices] = topk_values
        
        return S_sparse
    
    def compute_modularity(A, community_labels):

    # 图中节点的数量
        n = A.shape[0]
        
        # 计算图中的总边数 m
        m = np.sum(A) / 2
        
        # 计算每个节点的度数 k
        degrees = np.sum(A, axis=1)
        
        # 计算模块度
        Q = 0
        for i in range(n):
            for j in range(n):
                if community_labels[i] == community_labels[j]:
                    # A_ij - (k_i * k_j) / 2m
                    Q += A[i, j] - (degrees[i] * degrees[j]) / (2 * m)
        
        # 模块度公式：Q / 2m
        Q = Q / (2 * m)
        return Q
        
    def pretrain(self, module):

        print(module + ' Pretrain')

        for i in tqdm(range(len(self.layers)), desc="Layers trained: ", leave=True):
                self.setup_z(i, module)
                U, V = self.sklearn_pretrain(i)
                name = module + str(i)
                self.U_init[name] = U
                self.V_init[name] = V

        #M1 = compute_modularity(self.A.detach().cpu().numpy(),self.L)
        #M2 = compute_modularity(self.adjdrop.detach().cpu().numpy(),self.L)

        #print(M1, M2)

        with open(self.pretrain_params_path, 'wb') as handle:
            pickle.dump([self.U_init, self.V_init], handle, protocol=pickle.HIGHEST_PROTOCOL)
def compute_modularity(A, community_labels):

# 图中节点的数量
    n = A.shape[0]
    
    # 计算图中的总边数 m
    m = np.sum(A) / 2
    
    # 计算每个节点的度数 k
    degrees = np.sum(A, axis=1)
    
    # 计算模块度
    Q = 0
    for i in range(n):
        for j in range(n):
            if community_labels[i] == community_labels[j]:
                # A_ij - (k_i * k_j) / 2m
                Q += A[i, j] - (degrees[i] * degrees[j]) / (2 * m)
    
    # 模块度公式：Q / 2m
    Q = Q / (2 * m)
    return Q
        


