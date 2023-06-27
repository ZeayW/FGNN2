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
        'NOT': 0,
        'AND': 1,
        'PI': 2,
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

    graph.ndata["ntype"] = nodes_type
    graph.ndata['ntype2'] = th.argmax(nodes_type, dim=1).squeeze(-1)
    graph.ndata['output'] = th.zeros(graph.number_of_nodes(), dtype=th.float)
    graph.ndata['output'][topo[-1]] = 1


    return graph, topo

def test():
    g1, topo1 = parse_single_file("../../yosys/aigs/i4/aug1/i4_v9868_1.bench",4)
    # parse_single_file("../../yosys/aigs/i4/aug1/i4_v9868_2.bench",4)

    g1.ndata['temp'] = th.ones(size=(g1.number_of_nodes(), 32),
                                                       dtype=th.float)
    g1.ndata['h'] = th.ones((g1.number_of_nodes(), 32), dtype=th.float)
    from FunctionConv import FuncConv
    model = FuncConv(
                ntypes=2,
                hidden_dim=32,
                out_dim = 32
            )
    embeddings = model(g1,topo1,None)
    print(embeddings)
    for i,nids in enumerate(topo1):
        print('topo_level {}, nids: {}'.format(i,nids))
        print('\t embeddings:',embeddings[nids])


if __name__ == "__main__":
    options = get_options()

    per2replace = options.per2replace
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
        os.makedirs(os.path.join(save_path, 'i{}'.format(options.num_input)), exist_ok=True)
        if os.path.exists(data_file) is False:
            data_path = "{}/i{}/".format(rawdata_path,options.num_input)
            th.multiprocessing.set_sharing_strategy('file_system')

            orign_graphs = []
            positive_pairs = [[], [], [],[]]
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
                value = value[1:]
                required_input = int(required_input[1:])
                code = bin(int(value, 10))[2:].zfill(pow(2, required_input))

                original_graph, original_topo = parse_single_file(aig_file_path,required_input)
                if original_graph is None:
                    continue
                orign_graphs.append((original_graph, original_topo,code))

                for j in range(len(per2replace)):
                    positive_pair = [None, None]

                    for i in [1,2]:
                        augAig_file_path = os.path.join(
                            os.path.join(data_path,'aug{}'.format(per2replace[j])), '{}_{}.bench'.format(aig_name,i)
                        )
                        if not os.path.exists(augAig_file_path):
                            break
                        aug_graph, aug_topo = parse_single_file(augAig_file_path,required_input)
                        aug_graph.ndata['v'] = int(value) * th.ones(aug_graph.number_of_nodes(), dtype=th.float)
                        positive_pair[i-1] = [code,aug_graph, aug_topo[-1].item(),
                                            aug_graph.number_of_nodes()]  # gr` aph, PO, size
                    if None in positive_pair:
                        continue
                    positive_pairs[j].append(positive_pair)
            #print(positive_pairs[0])
            # print(dataset.batch_graph.ndata)

            save_file = os.path.join(save_path, 'i{}/origin.pkl'.format(options.num_input))
            with open(save_file, 'wb') as f:
                pickle.dump(orign_graphs, f)
            for j in range(len(per2replace)):
                save_file = os.path.join(save_path, 'i{}/aug{}.pkl'.format(options.num_input, per2replace[j]))
                with open(save_file, 'wb') as f:
                    print('i{}_aug{}, #pair:{}'.format(options.num_input,per2replace[j],len(positive_pairs[j])))
                    pickle.dump(positive_pairs[j], f)
