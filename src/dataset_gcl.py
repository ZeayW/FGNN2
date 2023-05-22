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


def parse_single_file(nodes,edges,output_node):
    # nodes: list of (node, {"type": type}) here node is a str ,like 'n123' or '1'b1'
    # note that here node type does not include buf /not
    label2id = {"1'b0": 0, "1'b1": 1, 'DFF': 2, 'DFFSSR': 3, 'DFFAS': 4, 'NAND': 5, 'AND': 6,
                'OR': 7, 'DELLN': 8, 'INV': 9, 'NOR': 10, 'XOR': 11, 'MUX': 12, 'XNOR': 13,
                'MAJ': 14, 'PI': 15}
    #print(nodes)
    #print(edges)
    nid = 0
    node2id = {}
    id2node = {}
    output_nid = None
    ntype = th.zeros((len(nodes), len(label2id.keys())), dtype=th.float)
    for n in nodes:
        if node2id.get(n[0]) is None:
            node2id[n[0]] = nid
            id2node[nid] = n[0]
            type = n[1]["type"]
            if re.search("\d", type):
                type = type[: re.search("\d", type).start()]
            ntype[nid][label2id[type]] = 1
            if n[0] == output_node:
                output_nid = nid
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
    PIs = th.tensor(range(graph.number_of_nodes()))[th.argmax(ntype,dim=1).squeeze(-1)==15]\
        .numpy().tolist()
    #if len(PIs)!=get_options().num_input:
    #print(PIs,output_nid)
    #assert len(PIs)==get_options().num_input
    #print(graph.nodes())
    depth = cal_depth(graph,PIs,output_nid)
    return graph,output_nid,depth

    
class Dataset_gcl(DGLDataset):
    def __init__(self,datapath):
        self.datapath = datapath
        super(Dataset_gcl, self).__init__(name="dac")
        # self.alpha = Parameter(th.tensor([1]))

    def process(self):
        type2label = {"ling_adder":1,"hybrid_adder":2, "cond_sum_adder":3, "sklansky_adder":4, "brent_kung_adder":5, "bounded_fanout_adder":6,"unknown":7}
        self.graphs = []
        self.len = 0
        max_depth = 0
        options = get_options()
        start_nid = 0
        self.POs = []

        self.aug_graphs = [[],[],[]]
        self.len = 0
        aug_max_depth = [0,0,0]

        start_nid = 0
        aug_start_nids = [0,0,0]
        self.aug_POs = [[],[],[]]
        # with open(os.path.join(self.datapath,'split{}.pkl'.format(self.split)),'rb') as f:
        #     filelist = pickle.load(f)
        # print(len(filelist))
        filelist = os.listdir(self.datapath)
        print(len(filelist))
        # print('file list{} start with {}'.format(self.split,filelist[0]))
        for vf in filelist:
            if not vf.endswith('.v') or not os.path.exists(os.path.join(self.datapath,vf)):
                continue
            #PO = []
            print('\ngenerate positive samples for {}'.format(vf))
            value = vf.split('_')[2].split('.')[0][1:]
            parser = DcParser('i{}_v{}'.format(options.num_input, value))
            output_node, nodes, edges = parser.parse(os.path.join(self.datapath, vf))
            if len(nodes) == 0:
                print('empty...')
                continue
            original_graph, original_PO, original_depth = parse_single_file(nodes, edges, output_node)
            self.POs.append(original_PO+start_nid)
            #PO.append((original_PO+start_nid,original_depth))
            self.graphs.append(original_graph)
            start_nid += original_graph.number_of_nodes()
            #print('depth:',original_depth)
            if original_depth>max_depth:
                max_depth = original_depth
            for num2replace in range(1,4):
                for i in range(2):
                    print('generating positive sample{}, num replaced = {}'.format(i,num2replace))
                    new_nodes, new_edges,output_nid = transform(nodes, edges, output_node,num2replace,options)
                    new_graph, new_PO, new_depth = parse_single_file(new_nodes, new_edges, output_nid)
                    self.aug_POs[num2replace-1].append(new_PO+aug_start_nids[num2replace-1])
                    self.aug_graphs[num2replace-1].append(new_graph)
                    aug_start_nids[num2replace-1] += new_graph.number_of_nodes()
                    if new_depth>aug_max_depth[num2replace-1]:
                        aug_max_depth[num2replace-1] = new_depth
                    #print('depth:',new_depth)
        self.depth = max_depth
        self.aug_depth = aug_max_depth
        self.batch_graph = dgl.batch(self.graphs)
        self.aug_batch_graphs =  []
        for i in range(3):
            self.aug_batch_graphs.append(dgl.batch(self.aug_graphs[i]))

    def __len__(self):
        return self.len


def change_order(nids,width):
    res = []
    if len(nids) % width !=0 :
        print('num of IO is wrong')
        assert False
    num_modules = int(len(nids) / width)
    for bit_index in range(width):
        for module_index in range(num_modules):
            res.append(nids[width*module_index+bit_index])
    return res
def cal_depth(g,PIs,output_nid):
    g = g.to_networkx()
    depth = 0
    dst = output_nid
    for src in PIs:
        #print('src',id2nodes[src])
        max_path_length = 0
        #for dst in PIs[2*i:2*i+2]:
        path = None
        if nx.has_path(g, src, dst):
            for p in nx.all_simple_paths(g, src, dst):
                if len(p) > max_path_length:
                    max_path_length = len(p)
                    path = p
        #print('src:{},dst:{},path:{}'.format(src,dst,path))
        depth = max(depth,max_path_length-1)

    return depth


if __name__ == "__main__":
    random.seed(726)
    datapaths = ["../dc/sub/implementation/test/"]

    th.multiprocessing.set_sharing_strategy('file_system')
    # print(dataset_not_edge.Dataset_n)
    dataset = Dataset_sub("adder", 32,datapaths, None)

    # graph = parse_single_file('../util1/adder4_d0.00_auto_adder.v')
    # print(graph)
