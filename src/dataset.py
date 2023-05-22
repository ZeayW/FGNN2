import sys

sys.path.append("..")

import dgl
from dgl.data import DGLDataset
import networkx as nx
import torch as th

import os
import random
from verilog_parser import DcParser
from options import get_options


def parse_single_file(parser,vfile_pair,hier_report):
    r"""

    generate the DAG for a circuit design

    :param parser: DCParser
        the parser used to transform neetlist to DAG
    :param vfile_pair: (str,str)
        the netlists for current circuit, including a hierarchical one and a non-hierarchical one
    :param hier_report: str
        the report file for current circuit
    :return: dglGraph
        the result DAG
    """

    # gate types
    label2id = {"1'b0": 0, "1'b1": 1, 'DFF': 2, 'DFFSSR': 3, 'DFFAS': 4, 'NAND': 5, 'AND': 6,
                 'OR': 7, 'DELLN': 8, 'INV': 9, 'NOR': 10, 'XOR': 11, 'MUX': 12, 'XNOR': 13,
                'MAJ': 14, 'PI': 15}


    nodes, edges = parser.parse(vfile_pair,hier_report)

    # build the dgl graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # assign an id to each node
    node2id = {}
    for n in nodes:
        if node2id.get(n[0]) is None:
            nid = len(node2id)
            node2id[n[0]] = nid

    # init the label tensors
    is_adder = th.zeros((len(node2id), 1), dtype=th.long)
    is_adder_input = th.zeros((len(node2id), 1), dtype=th.long)
    is_adder_output = th.zeros((len(node2id), 1), dtype=th.long)
    is_mul_input = th.zeros((len(node2id), 1), dtype=th.long)
    is_mul_output = th.zeros((len(node2id), 1), dtype=th.long)
    is_sub_input = th.zeros((len(node2id), 1), dtype=th.long)
    is_sub_output = th.zeros((len(node2id), 1), dtype=th.long)
    position = th.zeros((len(node2id), 1), dtype=th.long)

    # collect the label information
    for n in nodes:
        nid = node2id[n[0]]
        if get_options().region:
            is_adder[nid][0] = n[1]['is_adder']
        else:
            is_adder_input[nid][0] = n[1]["is_adder_input"]
            is_adder_output[nid][0] = n[1]["is_adder_output"]
            is_mul_input[nid][0] = n[1]["is_mul_input"]
            is_mul_output[nid][0] = n[1]["is_mul_output"]
            is_sub_input[nid][0] = n[1]["is_sub_input"]
            is_sub_output[nid][0] = n[1]["is_sub_output"]
            if n[1]["position"] is not None:
                position[nid][0] = n[1]["position"][1]

    # collect the node type information
    ntype = th.zeros((len(node2id), get_options().in_dim), dtype=th.float)
    for n in nodes:
        nid = node2id[n[0]]
        if label2id.get(n[1]['type']) is None:
            print('new type', n[1]['type'])
            if  'DFF' in n[1]['type']:
                ntype[nid][2] = 1
        else:
            ntype[nid][label2id[n[1]["type"]]] = 1

    # print('muldiv_outputs:',len(is_mul_output[is_mul_output==1]))
    # print('muldiv_inputs1:', len(is_mul_input[is_mul_input == 1]))
    # print('muldiv_inputs2:', len(is_mul_input[is_mul_input == 2]))
    # print('sub_outputs:', len(is_sub_output[is_sub_output == 1]))
    # print('sub_inputs1:', len(is_sub_input[is_sub_input == 1]))
    # print('sub_inputs2:', len(is_sub_input[is_sub_input == 2]))
    # print('adder_outputs:', len(is_adder_output[is_adder_output == 1]))
    # print('adder_inputs1:', len(is_adder_input[is_adder_input == 1]))

    src_nodes = []
    dst_nodes = []
    is_reverted = []

    for src, dst, edict in edges:
        src_nodes.append(node2id[src])
        dst_nodes.append(node2id[dst])
        is_reverted.append([0, 1] if edict["is_reverted"] else [1, 0])

    # create the graph
    graph = dgl.graph(
        (th.tensor(src_nodes), th.tensor(dst_nodes)), num_nodes=len(node2id)
    )

    graph.ndata["ntype"] = ntype

    print('ntype:',ntype.shape)

    # add label information
    graph.ndata['adder_i'] = is_adder_input
    graph.ndata['adder_o'] = is_adder_output
    graph.ndata['mul_i'] = is_mul_input
    graph.ndata['mul_o'] = is_mul_output
    graph.ndata['sub_i'] = is_sub_input
    graph.ndata['sub_o'] = is_sub_output

    graph.edata["r"] = th.FloatTensor(is_reverted)
    graph.ndata['position'] = position

    return graph


class Dataset(DGLDataset):
    def __init__(self, top_module,data_paths,report_folders,label2id):
        self.label2id =label2id
        self.data_paths = data_paths
        self.report_folders = report_folders
        self.parser = DcParser(top_module,adder_keywords=['add_x','alu_DP_OP','div_DP_OP'],sub_keywords=['sub_x'])
        super(Dataset, self).__init__(name="dac")

    def process(self):
        r"""

        transform the netlists to DAGs

        :return:

        """

        self.batch_graphs = []
        self.graphs = []
        self.len = 0
        vfile_pairs = {}
        for i,path in enumerate(self.data_paths):
            files = os.listdir(path)
            for v in files:
                if not v.endswith('v') or v.split('.')[0].endswith('d10') or 'auto' in v:
                    continue
                if v.startswith('hier'):
                    vname = v[5:-2]
                    vfile_pairs[vname] = vfile_pairs.get(vname, [])
                    vfile_pairs[vname].insert(0, v)
                else:
                    vname = v[:-2]
                    vfile_pairs[vname] = vfile_pairs.get(vname, [])
                    vfile_pairs[vname].append(v)
            vfile_pairs = vfile_pairs.values()
            # each circuit has 2 netlists: a hierarchical one and a non-hierarchical one
            for vfile_pair in vfile_pairs:
                hier_vf, vf = vfile_pair[0], vfile_pair[1]
                # the report file is also needed to label the target arithmetic blocks
                hier_report = os.path.join(self.report_folders[i], hier_vf[:-1] + 'rpt')
                hier_vf = os.path.join(path, hier_vf)
                vf = os.path.join(path, vf)

                print("Processing file {}".format(vfile_pair[1]))
                self.len += 1
                # parse single file
                graph = parse_single_file(self.parser, (hier_vf,vf), hier_report)

                self.graphs.append(graph)

        # combine all the graphs into a batch graph
        self.batch_graph = dgl.batch(self.graphs)

    def __len__(self):
        return self.len




