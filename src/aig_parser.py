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

sys.path.append("..")

options = get_options()

class MyLoader(th.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def parse_single_file(file_path,required_input):

    with open(file_path, 'r') as f:
        context = f.readlines()
    ntype2id = {
        'PI': 0,
        'NOT': 1,
        'AND': 2
    }
    nodes: List[Tuple[str, Dict[str, str]]] = []  # a list of (node, {"type": type})
    # edges: List[Tuple[str, str, Dict[str, bool]]] = []  # a list of (src, dst, {"is_reverted": is_reverted})
    num_input = 0
    src_nodes = []
    dst_nodes = []
    pin2nid = {}
    PO_pin = None
    for sentence in context[1:]:
        if len(sentence) == 0:
            continue
        if sentence.startswith('INPUT'):
            num_input += 1
            pin = sentence[sentence.find('(') + 1:sentence.rfind(')')]
            pin2nid[pin] = len(pin2nid)
            nodes.append((pin, {'ntype': 'PI', "is_PO": False}))
        elif sentence.startswith('OUTPUT'):
            PO_pin = sentence[sentence.find('(') + 1:sentence.rfind(')')]
        else:
            pin, expression = sentence.replace(' ', '').split("=")
            pin2nid[pin] = len(pin2nid)
            gate, parameters = expression.split("(")
            parameters = parameters.split(')')[0]
            nodes.append((pin, {'ntype': gate, 'is_PO': pin == PO_pin}))
            for input in parameters.split(','):
                # edges.append((input,pin,{}))
                src_nodes.append(pin2nid[input])
                dst_nodes.append(pin2nid[pin])
            # else:
            #     assert False, "wrong gate type: {}".format(gate)

    if num_input != required_input:
        return None, None

    nodes_type = th.zeros((len(nodes), len(ntype2id)))
    for node in nodes:
        nid = pin2nid[node[0]]
        ntypeID = ntype2id[node[1]['ntype']]
        nodes_type[nid][ntypeID] = 1

    graph = dgl.graph(
        (th.tensor(src_nodes), th.tensor(dst_nodes)), num_nodes=len(pin2nid)
    )

    topo = dgl.topological_nodes_generator(graph)

    return graph, topo


if __name__ == "__main__":
    options = get_options()

    save_path = options.datapath
    rawdata_path = options.rawdata_path
    if not os.path.exists(os.path.join(save_path, 'i{}'.format(options.num_input))):
        os.makedirs(os.path.join(save_path, 'i{}'.format(options.num_input)))
    stdout_f = os.path.join(save_path, "stdout.log")
    stderr_f = os.path.join(save_path, "stderr.log")
    with tee.StdoutTee(stdout_f), tee.StderrTee(stderr_f):
        print(options)
        th.multiprocessing.set_sharing_strategy('file_system')
        device = th.device("cuda:" + str(options.gpu) if th.cuda.is_available() else "cpu")

        data_file = os.path.join(save_path, 'i{}/origin.pkl'.format(options.num_input))

        if os.path.exists(data_file) is False:
            data_path = "{}/i{}/".format(rawdata_path,options.num_input)
            th.multiprocessing.set_sharing_strategy('file_system')

            orign_graphs = []
            positive_pairs = [[], [], []]
            #filelist = os.listdir(rawdata_path)

            #print('#cases:', len(filelist)
            for aig_file in os.listdir(data_path):
                if not aig_file.endswith('bench'):
                    continue
                aig_name = aig_file.split('.')[0]
                aig_file_path = os.path.join(data_path,aig_file)
                # if not vf.endswith('.v') or not os.path.exists(os.path.join(rawdata_path, vf)):
                #     continue
                required_input, value = aig_name.split('_')
                required_input = int(required_input[1:])
                code = bin(int(value[1:], 10))[2:].zfill(pow(2, required_input))

                original_graph, original_topo = parse_single_file(aig_file_path,required_input)
                if original_graph is None:
                    continue
                orign_graphs.append((original_graph, original_topo,code))

                for per2replace in [1,2,3,5]:
                    positive_pair = [None, None]

                    for i in range(2):
                        augAig_file_path = os.path.join(
                            os.path.join(data_path,'aug{}'.format(per2replace)), '{}_{}.bench'.format(aig_name,i)
                        )
                        if not os.path.exists(augAig_file_path):
                            break
                        aug_graph, aug_topo = parse_single_file(augAig_file_path,required_input)
                        aug_graph.ndata['v'] = int(value) * th.ones(aug_graph.number_of_nodes(), dtype=th.float)
                        aug_graph.ndata['output'] = th.zeros(aug_graph.number_of_nodes(), dtype=th.float)
                        aug_graph.ndata['output'][aug_topo[-1]] = 1
                        positive_pair[i] = [code,aug_graph, aug_topo[-1].item(),
                                            aug_graph.number_of_nodes()]  # graph, PO, size
                    if None in positive_pair:
                        continue
                    positive_pairs[per2replace - 1].append(positive_pair)
            print(positive_pairs[0])
            # print(dataset.batch_graph.ndata)

            save_file = os.path.join(save_path, 'i{}/origin.pkl'.format(options.num_input))
            with open(save_file, 'wb') as f:
                pickle.dump(orign_graphs, f)
            for j in [1,2,3,5]:
                save_file = os.path.join(save_path, 'i{}/aug{}.pkl'.format(options.num_input, j ))
                with open(save_file, 'wb') as f:
                    pickle.dump(positive_pairs[j], f)
