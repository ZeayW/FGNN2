"""Torch Module for FunctionConv layer"""
import dgl
import torch
from torch import nn
from dgl import function as fn

class MLP(torch.nn.Module):
    def __init__(self, *sizes, batchnorm=False, dropout=False):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(torch.nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(torch.nn.LeakyReLU(negative_slope=0.01))
                #fcs.append(torch.nn.ReLU())
                if dropout: fcs.append(torch.nn.Dropout(p=0.2))
                if batchnorm: fcs.append(torch.nn.BatchNorm1d(sizes[i]))
        self.layers = torch.nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)


class FuncConv(nn.Module):

    def __init__(self,
                 hidden_dim,
                 out_dim,
                 flag_proj=False,
                 activation=None):
        super(FuncConv, self).__init__()

        # initialize the gate functions, each for one gate type, e.g., AND, OR, XOR...

        self.func_inv = MLP(hidden_dim,int(hidden_dim/2),int(hidden_dim/2),hidden_dim)
        self.func_and = MLP(hidden_dim,int(hidden_dim/2),int(hidden_dim/2),hidden_dim)
        # set some attributes
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation = activation
        self.flag_proj = flag_proj
        if flag_proj: self.proj = MLP(hidden_dim,int(hidden_dim/2),int(hidden_dim/2),out_dim)
        # initialize the parameters
        #self.reset_parameters()

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


    def apply_nodes_func(self,nodes):

        res =  self.func_and(nodes.data['neigh'])
        # mask = nodes.data['inv'].squeeze()==1
        # res[mask] = self.func_inv((res[mask]))
        return {'rst':res}

    def edge_msg(self,edges):
        msg = edges.src['h']
        mask = edges.data['r'] == 1
        msg[mask] = self.func_inv(msg[mask])

        return {'m': msg}

    def forward(self,act_flag, graph, feat):
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

        with graph.local_scope():
            feat_src = feat

            graph.srcdata['h'] = feat_src
            r"""
            update_all is used to reduce and aggregate the messages 
            parameters:
                message_func (dgl.function.BuiltinFunction or callable) 
                    The message function to generate messages along the edges. 
                    It must be either a DGL Built-in Function or a User-defined Functions.
                reduce_func (dgl.function.BuiltinFunction or callable) 
                    The reduce function to aggregate the messages. 
                    It must be either a DGL Built-in Function or a User-defined Functions.
                apply_node_func (callable, optional) 
                    An optional apply function to further update the node features after 
                    the message reduction. It must be a User-defined Functions.
            """
            # we used mean as the reduce function , and a self-defined function as the apply function
            #print(feat_src.shape)
            # graph.apply_edges(lambda edges: {'eh' : self.func_inv(edges.src['h'])},
            #                   graph.edata[dgl.EID][graph.edata['r'].squeeze()==1])
            # # print(len(graph.edata[dgl.EID]),graph.edata[dgl.EID])
            # # print(graph.srcdata['h'])
            # # print(graph.edata['eh'])
            # # print(graph.edata['r'])
            # graph.apply_edges(lambda edges: {'eh': edges.src['h']},
            #                   graph.edata[dgl.EID][graph.edata['r'].squeeze() == 0])
            # print(graph.edata['eh'])
            # print('node_inv',graph.dstdata['inv'])
            graph.update_all(self.edge_msg, fn.mean('m', 'neigh'),self.apply_nodes_func)
            rst = graph.dstdata['rst']
            #print(graph.ntypes)
            # graph.apply_nodes(lambda nodes: {'rst': self.funv_inv(nodes.data['rst'])},
            #                   graph.dstdata[dgl.NID][graph.dstdata['inv'].squeeze()==1],'_N')
            # if act_flag and self.activation is not None:
            #     rst = self.activation(rst)

            return rst
