"""Torch Module for FunctionConv layer"""

import torch as th
from torch import nn
from dgl import function as fn


class MLP(th.nn.Module):
    def __init__(self, *sizes, negative_slope=0.01, batchnorm=False, dropout=False):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(th.nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(th.nn.LeakyReLU(negative_slope=negative_slope))
                if dropout: fcs.append(th.nn.Dropout(p=0.01))
                if batchnorm: fcs.append(th.nn.BatchNorm1d(sizes[i]))
        self.layers = th.nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)

class FuncConv(nn.Module):

    def __init__(self,
                 hidden_dim,
                 out_dim,
                 flag_proj = False,
                 flag_inv = True,
                 activation=None):
        super(FuncConv, self).__init__()

        self.hidden_dim = hidden_dim
        self.out_dim =out_dim
        self.flag_proj = flag_proj
        self.flag_inv = flag_inv
        self.func_inv = MLP(hidden_dim,int(hidden_dim/2),int(hidden_dim/2),hidden_dim)
        self.func_and = MLP(hidden_dim,int(hidden_dim/2),int(hidden_dim/2),hidden_dim)
        self.activation = activation
        self.proj_head = MLP(hidden_dim,hidden_dim,out_dim,negative_slope=0)
        # initialize the parameters
        # self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        pass
        gain = nn.init.calculate_gain('relu')
        for i in range(self.ntypes):
            nn.init.xavier_uniform_(self.gate_functions[i].weight, gain=gain)
        if self.flag_proj:
            nn.init.xavier_uniform_(self.proj.weight, gain=gain)

    def set_flag_proj(self,flag):
        self.flag_proj = flag

    def apply_nodes_func(self,nodes):
        res = self.func_and(nodes.data['neigh'])
        if self.flag_inv:
            mask = nodes.data['inv'].squeeze()==1
            res[mask] = self.func_inv((res[mask]))
        return {'h':res}

    def edge_msg(self,edges):
        msg = edges.src['h']
        #mask = edges.data['r'].squeeze()==1
        mask = edges.data['r'] == 1
        msg[mask] = self.func_inv(msg[mask])

        return {'m':msg}

    def forward(self, *parameters, flag_usage):

        def forward_global(graph, topo, PO_mask):
            r"""
            :param graph:  dgl graph
            :param topo:   topological levels of the input graph
            :param PO_mask:  mask gives the PO nids
            :return:
            """
            with graph.local_scope():
                for i, nodes in enumerate(topo[1:]):
                    graph.pull(nodes, self.edge_msg, fn.mean('m', 'neigh'), self.apply_nodes_func)
                rst = graph.ndata['h']
                if PO_mask is not None:
                    rst = rst[PO_mask]
                if self.flag_proj:
                    rst = self.proj_head(rst)
                return rst

        def forward_local(blocks, features):
            r"""
            :param blocks: llist of dgl block
            :param features:  input features of the first block
            :return:
            """
            depth = len(blocks)
            h = features
            r"""
            The message passes through the blocks layer by layer, from the PIs of the blocks to the POs
            In each iteration, messages are only passed between two successive blocks
            """
            for i in range(depth):
                graph = blocks[i]
                with graph.local_scope():
                    graph.srcdata['h'] = h
                    graph.update_all(self.edge_msg, fn.mean('m', 'neigh'), self.apply_nodes_func)
                    h = graph.dstdata['h']
            if self.flag_proj:
                h = self.proj_head(h)
            return h.squeeze(1)

        if flag_usage == 'global':
            return forward_global(*parameters)
        elif flag_usage == "local":
            return forward_local(*parameters)
        else:
            assert False, 'wrong usage of FuncConv'
