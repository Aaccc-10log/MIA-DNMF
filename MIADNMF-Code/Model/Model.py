import os
import torch
import pickle
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, SAGEConv
 
# class GatedAttention(torch.nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         # 门控权重计算
#         self.gate = torch.nn.Sequential(
#             torch.nn.Linear(hidden_dim * 3, 3),  # 输入 h1, h2, h3 的拼接
#             torch.nn.Softmax(dim=-1)             # 输出 3 个权重
#         )
#         self._init_weights()

#     def _init_weights(self):
#         # torch.nn.init.xavier_uniform_(self.gate[0].weight)
#         # torch.nn.init.constant_(self.gate[0].bias, 1.0)
#         torch.nn.init.zeros_(self.gate[0].weight)  # 权重初始化为0
#         torch.nn.init.constant_(self.gate[0].bias, 0.0) 

#     def forward(self, h1: torch.Tensor, h2: torch.Tensor, h3: torch.Tensor):
#         # 拼接三个向量 [batch_size, hidden_dim * 3]
#         h1 = h1.t()
#         h2 = h2.t()
#         h3 = h3.t()
#         h_cat = torch.cat([h1, h2, h3], dim=-1)
        
#         # 计算门控权重 [batch_size, 3]
#         weights = self.gate(h_cat)
        
#         # 加权求和 [batch_size, hidden_dim]
#         h_fused = weights[:, 0:1] * h1 + weights[:, 1:2] * h2 + weights[:, 2:3] * h3
#         return h_fused.t()
class Attention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dim * 3, 3)

        # 初始化为平均权重 logit
        torch.nn.init.zeros_(self.linear.weight)  # 不考虑输入，靠bias决定输出
        torch.nn.init.constant_(self.linear.bias, 0.0)  # softmax([0,0,0]) = [1/3,1/3,1/3]

    def forward(self, h1: torch.Tensor, h2: torch.Tensor, h3: torch.Tensor):
        h1 = h1.t()
        h2 = h2.t()
        h3 = h3.t()
        h_cat = torch.cat([h1, h2, h3], dim=-1)  # shape: [batch_size, hidden_dim * 3]

        # gate权重 shape: [batch_size, 3]
        logits = self.linear(h_cat)
        weights = F.softmax(logits, dim=-1)

        # 每个向量乘上对应的权重
        h_fused = weights[:, 0:1] * h1 + weights[:, 1:2] * h2 + weights[:, 2:3] * h3
        return h_fused.t(), weights
       

class MyModel(torch.nn.Module):
    # def __init__(self, encoder:Encoder, atten: GatedAttention, config):
    def __init__(self, atten: Attention, config):

        super(MyModel, self).__init__()
        self.config = config
        self.A = config['A']
        self.tau = config['tau']
        self.neg = config['neg']
        self.reg = config['reg']
        self.aloss = config['aloss']
        self.qloss = config['qloss']
        self.device = config['device']
        self.contra1 = config['contra1']
        self.classes = config['classes']
        self.is_init = config['is_init']
        self.netshape = config['netshape']
        self.num_nodes = config['num_nodes']
        self.pretrain_params_path = config['pretrain_params_path']
        self.feature = torch.eye(self.num_nodes, self.num_nodes, device=self.device)

        # self.mask = self.generate_mask(0.1)
        # self.mask = torch.ones(self.num_nodes, self.classes, device=self.device)

        self.U = torch.nn.ParameterDict({})
        self.V = torch.nn.ParameterDict({})

        # self.encoder: Encoder = encoder
        self.atten: Attention = atten

        self.fc1 = torch.nn.Linear(self.netshape[-1], self.netshape[1])
        self.fc2 = torch.nn.Linear(self.netshape[1], self.netshape[0])

        if os.path.isfile(self.pretrain_params_path):
            with open(self.pretrain_params_path, 'rb') as handle:
                self.U_init, self.V_init = pickle.load(handle)
        
        if self.is_init == True:

            module = 'origin'
            for i in range(len(self.netshape)):
                name = module + str(i)
                self.U[name] = torch.nn.Parameter(torch.tensor(self.U_init[name], dtype=torch.float32))
            self.V[name] = torch.nn.Parameter(torch.tensor(self.V_init[name], dtype=torch.float32))

            module = 'drop'
            for i in range(len(self.netshape)):
                name = module + str(i)
                self.U[name] = torch.nn.Parameter(torch.tensor(self.U_init[name], dtype=torch.float32))
            self.V[name] = torch.nn.Parameter(torch.tensor(self.V_init[name], dtype=torch.float32))

            module = 'global'
            for i in range(len(self.netshape)):
                name = module + str(i)
                self.U[name] = torch.nn.Parameter(torch.tensor(self.U_init[name], dtype=torch.float32))
            self.V[name] = torch.nn.Parameter(torch.tensor(self.V_init[name], dtype=torch.float32))

        else:
            print('need to init model')
            return None

    def forward(self):

        #######需要操作h1,h2,h3,注意力映射，hi->ReconA
        self.origin_V1 = self.V['origin' + str(len(self.netshape)-1)]
        self.drop_V1 = self.V['drop' + str(len(self.netshape)-1)]
        self.global_V1 = self.V['global' + str(len(self.netshape)-1)]

        h_fuse, weights = self.atten(self.origin_V1, self.drop_V1, self.global_V1)
        # print(self.origin_V1.shape, h_fuse.shape)

        return self.origin_V1, self.drop_V1, self.global_V1, h_fuse, weights
    
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z.t()))
        return self.fc2(z)
    
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())   
    
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor):

        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        loss = - torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

        return loss.mean()
    
    def compute_laplacian(self, A):

        D = torch.diag(A.sum(dim=1))
        return D - A
    
    def sigmoid_k(self, h, k, theta):

        return torch.sigmoid(k * (h - theta))
    
    def reconstructA(self, H, temper, theta):
        temp = torch.mm(H, H.T)
        max, min = temp.max(), temp.min()
        ReconA = (temp - min) / (max - min + 1e-8)
        ReconA = torch.sigmoid((ReconA - theta)/temper)
        return ReconA
    
    def loss_nonneg(self,module: str):

        loss = 0
        for i in range(len(self.netshape)):
            zero1 = torch.zeros_like(self.U[module + str(i)])
            X1 = torch.where(self.U[module + str(i)] > 0, zero1, self.U[module + str(i)])
            loss = loss + torch.square(torch.norm(X1))
        zero1 = torch.zeros_like(self.V[module + str(i)])
        X1 = torch.where(self.V[module + str(i)] > 0, zero1, self.V[module + str(i)])
        loss = loss + torch.square(torch.norm(X1))  

        return loss 
    def modularity(self, labels: torch.Tensor) -> float:
        A = self.A
        n = A.shape[0]
        m = A.sum() / 2
        if m == 0:
            return 0.0  
        k = A.sum(dim=1)
        B = A - torch.outer(k, k) / (2 * m)
        unique_labels = torch.unique(labels)
        num_communities = len(unique_labels)
        S = torch.zeros(n, num_communities, device=A.device)
        for i, label in enumerate(labels):
            S[i, label] = 1
        # 计算模块度 Q = (1/(2m)) * trace(S^T B S)
        Q = (S.T @ B @ S).trace() / (2 * m)
        return -Q.item()

    def decode(self, H: torch.Tensor):
        # 直接使用内积计算节点相似度
        adj_recon = torch.sigmoid(torch.mm(H, H.t()))
        # adj_recon = self.sigmoid_k(torch.sigmoid(torch.mm(H, H.t())), k=20, theta=0.5)*2 -1
        # adj_recon = self.reconstructA(H, 0.1, 0.2)
        return adj_recon - torch.eye(self.num_nodes, self.num_nodes, device=self.device)
    
    def decodeDNMF(self, h, w):
        def generateU(U, module):
            P1 = torch.eye(self.num_nodes, device=self.device)
            for i in range(len(self.netshape)):
                name = module + str(i)
                P1 = torch.mm(P1, U[name])
            return P1
        U1 = generateU(self.U, 'origin')
        U2 = generateU(self.U, 'drop')
        U3 = generateU(self.U, 'global')

        U_fuse = w[:,0:1]*U1 + w[:,1:2]*U2 + w[:,2:3]*U3
        ReconA = torch.mm(U_fuse, h)
        return ReconA  

    def loss(self, h, weights):

        # h1 = self.projection(self.origin_V1)
        # h2 = self.projection(self.drop_V1)
        # h3 = self.projection(self.global_V1)

        # h = self.projection(h)

        h1 = self.origin_V1
        h2 = self.drop_V1
        h3 = self.global_V1

        L = self.compute_laplacian(self.A)
        # print(h.shape)
        # X = torch.mm(h.t(), h)
        # ReconA = self.decode(h.t())
        ReconA = self.decodeDNMF(h, weights)
        loss1 = torch.square(torch.norm(self.A - ReconA)) 
        lossreg = torch.trace(h @ L @ h.t())
        loss1 = loss1 + self.reg*lossreg

        label = torch.argmax(h, dim=0)

        # lossQ = self.modularity(label)

        loss2 = self.contrastive_loss(h1, h)
        loss5 = self.loss_nonneg('origin') + self.loss_nonneg('drop') + self.loss_nonneg('global')

        loss = self.aloss * loss1 + self.neg* loss5 + self.contra1 * loss2

        return loss
    
        # lossreg = 0.001*torch.norm(ReconA, p=1)
        # loss1 = loss1 + lossreg
        # ReconL = self.compute_laplacian(ReconA)
        # lossLaplace = torch.square(torch.norm(L - ReconL))
        # loss1 = loss1 + lossLaplacez

        # criterion = torch.nn.BCELoss()
        # loss1 = criterion(ReconA, self.A)
        # P1 = torch.eye(self.num_nodes, device=self.device)
        # # print(P1)
        # # print(self.U['net0'])
        # for i in range(len(self.netshape)):
        #     P1 = torch.mm(P1, self.U['origin' + str(i)])
        # i = len(self.netshape) - 1
        # P1 = torch.mm(P1, self.V['origin' + str(i)])
        # loss1 = torch.square(torch.norm(self.A - P1))

        # loss2 = self.contrastive_loss(self.origin_V1, self.drop_V1)
        # loss3 = self.contrastive_loss(self.origin_V1, self.global_V1)