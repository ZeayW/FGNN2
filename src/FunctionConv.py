"""Torch Module for FunctionConv layer"""

import torch as th
from torch import nn
from dgl import function as fn

class Projection_Head(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=True,
                 activation=th.relu,
                 ):
        super(Projection_Head, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        hidden_feats = int(in_feats / 2)
        self.layers.append(nn.Linear(in_feats, hidden_feats, bias=bias))
        self.layers.append(nn.Linear(hidden_feats, out_feats, bias=bias))
        gain = nn.init.calculate_gain('relu')
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)

    def forward(self, features):
        h = features
        h = self.activation(self.layers[0](h))
        h = self.layers[1](h).squeeze(-1)

        return h

class MLP(th.nn.Module):
    def __init__(self, *sizes, batchnorm=False, dropout=False):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(th.nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(th.nn.LeakyReLU(negative_slope=0.01))
                if dropout: fcs.append(th.nn.Dropout(p=0.01))
                if batchnorm: fcs.append(th.nn.BatchNorm1d(sizes[i]))
        self.layers = th.nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)

class FuncConv(nn.Module):

    def __init__(self,
                 ntypes,
                 hidden_dim,
                 out_dim,
                 flag_proj = False,
                 activation=None):
        super(FuncConv, self).__init__()

        # initialize the gate functions, each for one gate type, e.g., AND, OR, XOR...
        self.hidden_dim = hidden_dim
        self.out_dim =out_dim
        self.flag_proj = flag_proj
        # self.gate_functions = nn.ModuleList()
        # for i in range(ntypes):
        #     self.gate_functions.append(
        #         MLP(hidden_dim,int(hidden_dim/2),int(hidden_dim/2),hidden_dim)
        #     )
            # self.gate_functions.append(nn.Linear(hidden_dim,hidden_dim))

        self.funv_inv = MLP(hidden_dim,int(hidden_dim/2),int(hidden_dim/2),hidden_dim)
        self.func_and = MLP(hidden_dim,int(hidden_dim/2),int(hidden_dim/2),hidden_dim)

        # set some attributes
        self.ntypes =ntypes
        #self._out_feats = out_feats
        self.activation = activation
        if flag_proj: self.proj = MLP(hidden_dim,int(hidden_dim/2),int(hidden_dim/2),out_dim)
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

    def apply_nodes_func(self,nodes):
        res = self.func_and(nodes.data['neigh'])
        # mask = nodes.data['inv'].squeeze()==1
        # res[mask] = self.funv_inv((res[mask]))
        return {'h':res}

    def edge_msg(self,edges):
        msg = edges.src['h']
        mask = edges.data['r'].squeeze()==1
        msg[mask] = self.func_inv(msg[mask])

        return {'m':msg}

    def forward(self, graph, topo,PO_mask):
        r"""

        Description
        -----------
        Compute FunctionalGNN layer.

        Parameters
        ----------
        act_flag: boolean
            determine whether to use an activation function or not
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        #print(graph.ndata['h'])
        with graph.local_scope():
            for i, nodes in enumerate(topo[1:]):
                graph.pull(nodes, self.edge_msg, fn.mean('m', 'neigh'), self.apply_nodes_func)

            rst = graph.ndata['h']
            if PO_mask is not None:
                rst = rst[PO_mask]
            # if self.activation is not None:
            #     rst = self.activation(rst)
            if self.flag_proj:
                rst = self.proj(rst)
            return rst
