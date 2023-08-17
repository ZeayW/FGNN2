import dgl
from dgl.data import DGLDataset
import networkx as nx
import torch as th
import numpy as np
import os
import random
from options import get_options
from torch.nn.parameter import Parameter
import pickle
import tee
import sys
from truthvalue_parser import *

sys.path.append("..")

options = get_options()


def parse_single_file(file_path):

    with open(file_path, 'r') as f:
        context = f.readlines()
    ntype2id = {
        'AND': 0,
        'PI': 1,
    }
    nodes = {}
    # edges: List[Tuple[str, str, Dict[str, bool]]] = []  # a list of (src, dst, {"is_reverted": is_reverted})
    num_input = 0
    src_nodes = []
    dst_nodes = []
    edges = []
    inv_map = {}
    pin2nid = {}
    PO_pins = []
    PO_nodes = []
    for sentence in context[1:]:
        if len(sentence) == 0:
            continue
        if sentence.startswith('INPUT'):
            num_input += 1
            pin = sentence[sentence.find('(') + 1:sentence.rfind(')')]
            nodes[pin] =  {'ntype': 'PI', 'inv':False}
        elif sentence.startswith('OUTPUT'):
            PO_pins.append(
                sentence[sentence.find('(') + 1:sentence.rfind(')')]
            )
        else:
            pin, expression = sentence.replace(' ', '').split("=")
            gate, parameters = expression.split("(")
            parameters = parameters.split(')')[0]
            if gate == 'NOT':
                predecessor = parameters
                inv_map[pin] = predecessor
                if pin in PO_pins:
                    nodes[predecessor]['inv'] = True
                    PO_nodes.append(predecessor)
            elif gate == 'AND':
                nodes[pin] =  {'ntype': gate, 'inv':False}
                if pin in PO_pins:
                    PO_nodes.append(pin)
                for input in parameters.split(','):
                    src = inv_map.get(input,input)
                    is_reverted = src!=input
                    edges.append(
                        (src,pin,{'r':is_reverted})
                    )
            else:
                assert False
            # for input in parameters.split(','):
                # edges.append((input,pin,{}))
                # src_nodes.append(pin2nid[input])
                # dst_nodes.append(pin2nid[pin])
            # else:
            #     assert False, "wrong gate type: {}".format(gate)

    n_inv = th.zeros((len(nodes), 1), dtype=th.float)
    nodes_type = th.zeros((len(nodes), len(ntype2id)))
    for pin,node_info in nodes.items():
        pin2nid[pin] = len(pin2nid)
        ntypeID = ntype2id[node_info['ntype']]
        nodes_type[pin2nid[pin]][ntypeID] = 1
        n_inv[pin2nid[pin]][0] = node_info['inv']

    PO_nids = [pin2nid[p] for p in PO_nodes ]

    e_reverted = th.zeros((len(edges), 1), dtype=th.long)
    for eid, (src, dst, edict) in enumerate(edges):
        src_nodes.append(pin2nid[src])
        dst_nodes.append(pin2nid[dst])
        e_reverted[eid][0] = edict['r']


    graph = dgl.graph(
        (th.tensor(src_nodes), th.tensor(dst_nodes)), num_nodes=len(pin2nid)
    )
    topo = dgl.topological_nodes_generator(graph)

    graph.ndata["ntype"] = nodes_type
    #graph.ndata['ntype2'] = th.argmax(nodes_type, dim=1).squeeze(-1)
    graph.ndata['output'] = th.zeros(graph.number_of_nodes(), dtype=th.float)
    graph.ndata['output'][topo[-1]] = 1
    graph.ndata['inv'] = n_inv.squeeze()

    graph.edata['r'] = e_reverted.squeeze()

    print(PO_nodes)
    print(graph.ndata['inv'][PO_nids])
    return graph, topo, PO_nids



if __name__ == "__main__":
    options = get_options()

    save_path = options.datapath
    os.makedirs(save_path,exist_ok=True)
    rawdata_path = options.rawdata_path
    stdout_f = os.path.join(save_path, "stdout.log")
    stderr_f = os.path.join(save_path, "stderr.log")


    block2label = {
        'adder':0,
        'multiplier':1,
        'divider':2,
        'subtractor':3
    }
    targets = ['train','test']
    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
        print(options)
        th.multiprocessing.set_sharing_strategy('file_system')

        for target in targets:
            target_dir = os.path.join(rawdata_path,target)
            for block in os.listdir(target_dir):
                dataset = []
                label = block2label.get(block,None)
                if label is None:
                    continue
                netlists_dir = os.path.join(target_dir,block)
                for netlist in os.listdir(netlists_dir):
                    netlist_file_path = os.path.join(netlists_dir,netlist)
                    if not os.path.isfile(netlist_file_path):
                        continue
                    graph, topo,PO_nids = parse_single_file(netlist_file_path)
                    print(netlist_file_path)
                    print(graph,PO_nids)
                    exit()
                    dataset.append((graph,topo,PO_nids,label))
                os.makedirs( os.path.join(save_path,target),exist_ok=True)
                save_file = os.path.join(save_path,target,'{}.pkl'.format(block))
                with open(save_file, 'wb') as f:
                    print('{}, #netlist:{}'.format(block,len(dataset)))
                    pickle.dump(dataset, f)
