"""Torch Module for FunctionConv layer"""

import torch
from torch import nn
from dgl import function as fn


class FunctionConv(nn.Module):

    def __init__(self,
                 ntypes,
                 in_feats,
                 out_feats,
                 activation=None):
        super(FunctionConv, self).__init__()

        # initialize the gate functions, each for one gate type, e.g., AND, OR, XOR...
        self.gate_functions = nn.ModuleList()
        for i in range(ntypes):
            self.gate_functions.append(nn.Linear(in_feats,out_feats))

        # set some attributes
        self.ntypes =ntypes
        self._out_feats = out_feats
        self.activation = activation

        # initialize the parameters
        self.reset_parameters()

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
        r"""

               Description
               -----------
               An apply function to further update the node features after the message reduction.

               Parameters
               ----------
               nodes: the applied nodes

               Returns
               -------
               {'msg_name':msg}
               msg: torch.Tensor
                   The aggregated messages of shape :math:`(N, D_{out})` where : N is number of nodes, math:`D_{out}`
                   is size of messages.
               """

        gate_inputs = nodes.data['neigh']   # the messages to aggregate
        gate_types = nodes.data['ntype2']   # the node-type of the target nodes
        res = nodes.data['temp']            # a tensor used to save the result messages

        # for each gate type, use an independent aggregator (function) to aggregate the messages
        for i in range(self.ntypes):
            mask = gate_types==i    # nodes of type i
            #print(self.gate_functions[i].weight.shape, gate_inputs[mask].shape,self.gate_functions[i](gate_inputs[mask]))
            res[mask] = self.gate_functions[i](gate_inputs[mask])

        return {'rst':res}

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
            graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'),self.apply_nodes_func)
            rst = graph.dstdata['rst']
            if act_flag and self.activation is not None:
                rst = self.activation(rst)

            return rst
