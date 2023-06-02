import sys

sys.path.append("..")

import dgl
from dgl.data import DGLDataset
import networkx as nx
import torch as th
import numpy as np
import os
import random
from single_verilog_parser import *
from options import get_options
from torch.nn.parameter import Parameter
from generate_data import *
import pickle

options = get_options()

def parse_single_file(nodes,edges,output_node):
    # nodes: list of (node, {"type": type}) here node is a str ,like 'n123' or '1'b1'
    # note that here node type does not include buf /not
    label2id = {'NAND': 0, 'AND': 1,
                'OR': 2,  'INV': 3, 'NOR': 4, 'XOR': 5, 'MUX': 6, 'XNOR': 7,
                'MAJ': 8, 'PI': 9}
    #print(nodes)
    #print(edges)
    nid = 0
    node2id = {}
    id2node = {}
    ntype = th.zeros((len(nodes), len(label2id.keys())), dtype=th.float)
    for n in nodes:
        if node2id.get(n[0]) is None:
            node2id[n[0]] = nid
            id2node[nid] = n[0]
            type = n[1]["type"]
            if re.search("\d", type):
                type = type[: re.search("\d", type).start()]
            ntype[nid][label2id[type]] = 1
            nid += 1

    src_nodes = []
    dst_nodes = []
    for src, dst, edict in edges:
        src_nodes.append(node2id[src])
        dst_nodes.append(node2id[dst])
    #print(src_nodes,dst_nodes)
    graph = dgl.graph(
        (th.tensor(src_nodes), th.tensor(dst_nodes)), num_nodes=len(node2id)
    )
    graph.ndata["ntype"] = ntype

    topo = dgl.topological_nodes_generator(graph)

    return graph,topo


class Aug_graph:
    def __init__(self,graph,topo,PO,size):
        self.graph = graph
        self.topo = topo
        self.PO =PO
        self.size = size

if __name__ == "__main__":
    options = get_options()
    th.multiprocessing.set_sharing_strategy('file_system')
    device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")
    save_path = options.datapath
    if not os.path.exists(os.path.join(save_path, 'i{}'.format(options.num_input))):
        os.makedirs(os.path.join(save_path, 'i{}'.format(options.num_input)))
    data_file = os.path.join(save_path, 'i{}/origin.pkl'.format(options.num_input))

    if os.path.exists(data_file) is False:
        rawdata_path = "../truthtables/i{}/implementation/".format(options.num_input)
        th.multiprocessing.set_sharing_strategy('file_system')

        orign_graphs = []
        positive_pairs = [[], [], []]
        filelist = os.listdir(rawdata_path)
        print('#cases:',len(filelist))
        for vf in filelist:
            if not vf.endswith('.v') or not os.path.exists(os.path.join(rawdata_path, vf)):
                continue
            print('\ngenerate positive samples for {}'.format(vf))
            value = vf.split('_')[2].split('.')[0][1:]
            parser = DcParser('i{}_v{}'.format(options.num_input, value))
            output_node, nodes, edges = parser.parse(os.path.join(rawdata_path, vf))
            if len(nodes) == 0:
                print('empty...')
                continue
            original_graph, original_topo = parse_single_file(nodes, edges, output_node)
            orign_graphs.append((original_graph, original_topo))

            for num2replace in range(1, 4):
                positive_pair = [None,None]
                for i in range(2):
                    print('generating positive sample{}, num replaced = {}'.format(i, num2replace))
                    new_nodes, new_edges, output_nid = transform(nodes, edges, output_node, num2replace, options)
                    new_graph, new_topo = parse_single_file(new_nodes, new_edges, output_nid)
                    positive_pair[i] = Aug_graph(new_graph, new_topo, new_topo[-1].item(),
                                                 new_graph.number_of_nodes()) # graph, topo_levels, PO, size
                    positive_pairs[num2replace - 1].append(positive_pair)
        # print(dataset.batch_graph.ndata)
        save_file = os.path.join(save_path, 'i{}/origin.pkl'.format(options.num_input))
        with open(save_file, 'wb') as f:
            pickle.dump(orign_graphs, f)
        for j in range(3):
            save_file = os.path.join(save_path, 'i{}/aug{}.pkl'.format(options.num_input, j + 1))
            with open(save_file, 'wb') as f:
                pickle.dump(positive_pairs[j], f)
