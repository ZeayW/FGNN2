import torch as th
import torch.nn as nn
import torch.nn.functional as F

#from MyGAT import GATConv
from MySageConv import SAGEConv
from FunctionConv3 import FunctionConv
#from MyGIN import GINConv

from time import time



class FuncGNN(nn.Module):
    def __init__(
        self,
        ntypes,
        hidden_dim,
        out_dim,
        dropout,
        n_layers = None,
        in_dim=16,
        activation=th.relu,

    ):
        super(FuncGNN, self).__init__()
        self.activation = activation
        self.ntypes = ntypes
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(p=dropout)
        self.fc_out = nn.Linear(hidden_dim,out_dim)

        self.conv = FunctionConv(
                    ntypes,
                    hidden_dim,
                    hidden_dim,
                    activation=activation,
                )


    def forward(self, blocks, features):
        r"""

        Description
        -----------
        forward computation of FGNN

        Parameters
        ----------
        blocks : [dgl_block]
            blocks gives the sampled neighborhood for a batch of target nodes.
            Given a target node n, its sampled neighborhood is organized in layers
            depending on the distance to n.
            A block is a graph that describes the part between two succesive layers,
            consisting of two sets of nodes: the *input* nodes and *output* nodes.
            The output nodes of the last block are the target nodes (POs), and
            the input nodes of the first block are the PIs.
        feature : torch.Tensor
            It represents the input (PI) feature of shape :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of target nodes (POs).

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output (PO) feature.
        """
        depth = len(blocks)
        h = features
        r"""
        The message passes through the blocks layer by layer, from the PIs of the blocks to the POs
        In each iteration, messages are only passed between two successive layers (blocks)
        """
        for i in range(depth):
            if i != 0:
                h = self.dropout(h)
            # we do not need activation function in the last iteration
            act_flag = (i != depth - 1)
            h = self.conv(act_flag, blocks[i], h)
        if self.hidden_dim!=self.out_dim:
            h= self.fc_out(h)
        return h.squeeze(1)


class ABGNN(nn.Module):
    def __init__(
        self,
        ntypes,
        hidden_dim,
        out_dim,
        dropout,
        n_layers=None,
        in_dim=16,
        activation=th.relu,
    ):
        super(ABGNN, self).__init__()
        self.activation = activation
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.fc_init = nn.Linear(in_dim,hidden_dim)
        in_dim = hidden_dim

        self.conv = SAGEConv(
            hidden_dim,
            hidden_dim,
            include=False,
            combine_type='sum',
            aggregator_type='mean',
            activation=activation,
        )

    def forward(self, blocks, features):
        r"""

                Description
                -----------
                forward computation of FGNN

                Parameters
                ----------
                blocks : [dgl_block]
                    blocks gives the sampled neighborhood for a batch of target nodes.
                    Given a target node n, its sampled neighborhood is organized in layers
                    depending on the distance to n.
                    A block is a graph that describes the part between two succesive layers,
                    consisting of two sets of nodes: the *input* nodes and *output* nodes.
                    The output nodes of the last block are the target nodes (POs), and
                    the input nodes of the first block are the PIs.
                feature : torch.Tensor
                    It represents the input (PI) feature of shape :math:`(N, D_{in})`
                    where :math:`D_{in}` is size of input feature, :math:`N` is the number of target nodes (POs).

                Returns
                -------
                torch.Tensor
                    The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
                    is size of output (PO) feature.
                """

        r"""
        The message passes through the blocks layer by layer, from the PIs of the blocks to the POs
        In each iteration, messages are only passed between two successive layers (blocks)
        """
        depth = len(blocks)
        h = self.activation(self.fc_init(features))
        for i in range(depth):
            if i != 0:
                h = self.dropout(h)
            act_flag = (i != depth - 1)
            #print(i,blocks[i])
            h = self.conv(act_flag, blocks[i], h)
        return h.squeeze(1)

class GraphSage(nn.Module):
    def __init__(
        self,
        ntypes,
        hidden_dim,
        out_dim,
        dropout,
        n_layers=None,
        in_dim=16,
        activation=th.relu,
    ):
        super(GraphSage, self).__init__()
        self.activation = activation
        #print(n_layers)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.fc_init = nn.Linear(in_dim,hidden_dim)
        in_dim = hidden_dim

        for i in range(n_layers):
            self.layers.append(
                SAGEConv(
                    in_dim,
                    hidden_dim,
                    include=True,
                    combine_type = 'sum',
                    aggregator_type='mean',
                    activation=activation,
                )
            )
            in_dim = hidden_dim
        # output layer

        self.layers.append(SAGEConv(in_dim, out_dim,include=True,combine_type='sum', aggregator_type='mean'))

    def forward(self, blocks, features):
        r"""

                Description
                -----------
                forward computation of FGNN

                Parameters
                ----------
                blocks : [dgl_block]
                    blocks gives the sampled neighborhood for a batch of target nodes.
                    Given a target node n, its sampled neighborhood is organized in layers
                    depending on the distance to n.
                    A block is a graph that describes the part between two succesive layers,
                    consisting of two sets of nodes: the *input* nodes and *output* nodes.
                    The output nodes of the last block are the target nodes (POs), and
                    the input nodes of the first block are the PIs.
                feature : torch.Tensor
                    It represents the input (PI) feature of shape :math:`(N, D_{in})`
                    where :math:`D_{in}` is size of input feature, :math:`N` is the number of target nodes (POs).

                Returns
                -------
                torch.Tensor
                    The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
                    is size of output (PO) feature.
                """
        h = self.activation(self.fc_init(features))
        for i in range(self.n_layers + 1):
            if i != 0:
                h = self.dropout(h)
            h = self.layers[i](True,blocks[i], h)

        return h.squeeze(1)

class Projection_Head(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=True,
                 activation=th.nn.functional.relu,
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

class MLP(nn.Module):
    def __init__(self,in_dim,out_dim,nlayers,activation =nn.ReLU() ,dropout=0.5):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.layers= nn.Sequential()
        dim1 = in_dim
        for i in range(nlayers-1):
            #self.layers.add_module('dropout_{}'.format(i+1),self.dropout)
            self.layers.add_module('linear_{}'.format(i+1),nn.Linear(dim1, int(dim1/2)))
            self.layers.add_module('activation_{}'.format(i + 1), self.activation)
            dim1 = int(dim1 / 2)
        self.layers.add_module('linear_{}'.format(nlayers),nn.Linear(dim1, out_dim))
    def forward(self,embedding):
        return self.layers(embedding).squeeze(-1)


class BiClassifier(nn.Module):
    def __init__(
        self, GCN1,GCN2,mlp
    ):
        super(BiClassifier, self).__init__()

        self.GCN1 = GCN1
        self.GCN2 = GCN2
        self.mlp=mlp
        # print(self.layers)
    def forward(self, in_blocks, in_features,out_blocks,out_features):
        if self.GCN2 is None:
            h = self.GCN1(in_blocks,in_features)
        elif self.GCN1 is None:
            h = self.GCN2(out_blocks, out_features)
        else:
            h = self.GCN1(in_blocks, in_features)
            rh = self.GCN2(out_blocks,out_features)
            h = th.cat((h,rh),1)
        h = self.mlp(h)

        return h