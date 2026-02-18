import numpy as np
import linecache
import torch


class Dataset(object):

    def __init__(self, config):
        self.graph_file = config['graph_file']
        self.label_file = config['label_file']
        self.device = config['device']

        self.A, self.L, self.num_classes, self.edge_index = self._load_data()
        self.num_nodes = self.A.shape[0]
        self.feature = torch.eye(self.num_nodes, self.num_nodes, device=self.device)

        self.num_edges = np.sum(np.triu(self.A, k=0))
        self.A = torch.tensor(self.A, dtype=torch.float32, device=self.device)
        self.L = torch.tensor(self.L, dtype=torch.float32, device=self.device)
        print('nodes {}, edes {}, classes {}'.format(self.num_nodes, self.num_edges, self.num_classes))


    def _load_data(self):

        lines = linecache.getlines(self.label_file)
        lines = [line.rstrip('\n') for line in lines]

        #===========load label============
        node_map = {}
        label_map = {}
        Y = []
        cnt = 0
        for idx, line in enumerate(lines):
            line = line.split(' ')
            node_map[line[0]] = idx
            y = []
            for label in line[1:]:
                if label not in label_map:
                    label_map[label] = cnt
                    cnt += 1
                y.append(label_map[label])
            Y.append(y)
        num_classes = len(label_map)
        num_nodes = len(node_map)

        L = np.array([la[0] for la in Y])

        headlist = []
        taillist = []

        #==========load graph========
        A = np.zeros((num_nodes, num_nodes))
        lines = linecache.getlines(self.graph_file)
        lines = [line.rstrip('\n') for line in lines]
        for line in lines:
            line = line.split(' ')
            idx1 = node_map[line[0]]
            idx2 = node_map[line[1]]
            A[idx2, idx1] = 1.0
            A[idx1, idx2] = 1.0
            headlist.append(idx1)
            taillist.append(idx2)
        
        edge_index = torch.tensor([headlist, taillist], dtype=torch.long, device=self.device)

        return A, L, num_classes, edge_index



