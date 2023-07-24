import sys

sys.path.append("..")

from verilog_parser import DcParser
from options import get_options

import networkx as nx
import os
import torch as th
import dgl
import pickle

lib = {}

def parse_single_cell(file_path):

    with open(file_path, 'r') as f:
        context = f.readlines()

    internal_nodes: List[Tuple[str, Dict[str, str]]] = []  # a list of (node, {"type": type})
    edges: List[Tuple[str, str, Dict[str, bool]]] = []  # a list of (src, dst, {"is_reverted": is_reverted})
    num_input = 0
    PIs = []
    PO = None
    PO_type = None
    pin2nid = {}
    for sentence in context[1:]:
        if len(sentence) == 0:
            continue
        if sentence.startswith('INPUT'):
            PIs.append(sentence[sentence.find('(') + 1:sentence.rfind(')')])
        elif sentence.startswith('OUTPUT'):
            PO = sentence[sentence.find('(') + 1:sentence.rfind(')')]
        else:
            pin, expression = sentence.replace(' ', '').split("=")
            pin2nid[pin] = str(len(pin2nid))
            gate, parameters = expression.split("(")
            parameters = parameters.split(')')[0]
            for i in parameters.split(','):
                input = pin2nid[i] if i not in PIs else i
                output = pin2nid[pin] if pin!=PO else "PO"
                edges.append((input,output,{}))
            if pin == PO:
                PO_type = gate
            else:
                internal_nodes.append((pin2nid[pin], {'type': gate, "is_adder_output": False, 'is_adder_input': False,
                                                      'position': None, 'is_mul_output': False, 'is_mul_input': False,
                                                      'is_sub_output':     False, 'is_sub_input': False, 'is_internal':True}))
            # else:
            #     assert False, "wrong gate type: {}".format(gate)


    return internal_nodes,edges, PO_type

def read_lib(lib_path):
    for cell_file in os.listdir(lib_path):
        if not cell_file.endswith('bench'):
            continue
        print(cell_file)
        cell_name = cell_file.split('.')[0]
        file_path = os.path.join(lib_path,cell_file)
        nodes,edges, PO_type = parse_single_cell(file_path)
        lib[cell_name.upper()] = (nodes,edges, PO_type)
    for cell, info in lib.items():
        print(cell, '#internal nodes: {}, #edges:{}'.format(len(info[0]),len(info[1])))

def graph2aig(g,buff_replace):
    print("transforming graph to aig...")
    new_nodes = []
    new_edges = []
    g_topo = nx.topological_sort(g)
    for i,n in enumerate(g_topo):
        n_property = g.nodes[n]
        n_property['is_internal'] = False
        predecessors = g.nodes[n]['inputs']
        cell_type = g.nodes[n]['type']
        if cell_type == "PI":
            new_nodes.append((
                n, n_property)
            )
            continue
        if (i%10000==0): print("\tprocessed: {}/{}".format(i,g.number_of_nodes()))
        added_nodes, added_edges, PO_type = lib[cell_type]
        n_property['type'] = PO_type
        new_nodes.append((
            n,n_property)
        )
        #print('root',n)
        for node in added_nodes:
            new_nodes.append(
                ("{}_{}".format(n, node[0]), node[1])
            )
            #print("\t{}_{}".format(n, node[0]))
        print(cell_type)
        for edge in added_edges:
            src, dst, e_property = edge
            dst = n if dst == "PO" else "{}_{}".format(n,dst)
            src = predecessors.get(src,"{}_{}".format(n,src))
            src = buff_replace.get(src,src)
            # if predecessors.get(src,None) is not None and g.nodes[src]["type"] == "NOT" and not g.nodes[src]["is_adder_output"] and e_property["type"] == "NOT":
            #     pre_p
            #print('\t ({}, {})'.format(src,dst))
            new_edges.append(
                (src,dst,{})
            )
            #g.add_edge((src,dst,{}))

    return new_nodes,new_edges

def aig2dglG(nodes,edges):
    ntype2id = {
        'NOT': 0,
        'AND': 1,
        'PI': 2,
    }

    # assign an id to each node
    target_nids = []
    node2id = {}
    for n in nodes:
        if node2id.get(n[0]) is None:
            nid = len(node2id)
            node2id[n[0]] = nid


    # init the label tensors
    is_internal = th.zeros((len(node2id), 1), dtype=th.long)
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
        is_internal[nid][0] = n[1]['is_internal']
        is_adder_input[nid][0] = n[1]["is_adder_input"]
        is_adder_output[nid][0] = n[1]["is_adder_output"]
        is_mul_input[nid][0] = n[1]["is_mul_input"]
        is_mul_output[nid][0] = n[1]["is_mul_output"]
        is_sub_input[nid][0] = n[1]["is_sub_input"]
        is_sub_output[nid][0] = n[1]["is_sub_output"]
        if n[1]["position"] is not None:
            #print(nid,n[0],n[1]["position"])
            position[nid][0] = n[1]["position"][1]

    # collect the node type information
    ntype = th.zeros((len(node2id), 3), dtype=th.float)
    for n in nodes:
        nid = node2id[n[0]]
        if ntype2id.get(n[1]['type']) is None:
            print('{}, unkown type:{} '.format(n[0],n[1]['type']))
        else:
            ntype[nid][ntype2id[n[1]["type"]]] = 1

    src_nodes = []
    dst_nodes = []
    is_reverted = []

    for src, dst, edict in edges:
        src_nodes.append(node2id[src])
        dst_nodes.append(node2id[dst])

    # create the graph
    graph = dgl.graph(
        (th.tensor(src_nodes), th.tensor(dst_nodes)), num_nodes=len(node2id)
    )

    dgl.topological_nodes_generator(graph)

    graph.ndata["ntype"] = ntype

    #print('ntype:',ntype.shape)
    #print(position[is_adder_output==1])
    # add label information
    graph.ndata['internal'] = is_internal
    graph.ndata['adder_i'] = is_adder_input
    graph.ndata['adder_o'] = is_adder_output
    graph.ndata['mul_i'] = is_mul_input
    graph.ndata['mul_o'] = is_mul_output
    graph.ndata['sub_i'] = is_sub_input
    graph.ndata['sub_o'] = is_sub_output

    graph.ndata['position'] = position

    print('# unmasked nodes: ',len(th.tensor(range(graph.number_of_nodes()))[graph.ndata['internal'].squeeze()==0]))
    return graph

if __name__ == "__main__":
    read_lib("../config/lib_aigs")
    options = get_options()
    save_path = options.datapath
    rawdata_path = options.rawdata_path
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    with open("../config/local_designs.txt",'r') as f:
        lines = f.readlines()
        design_list = [l.replace('\n', '') for l in lines if not l.startswith('#')]
        design_list = [d.split(" ") for d in design_list]

    for design, top_module in design_list:
        save_file = os.path.join(save_path, '{}.pkl'.format(design))
        if os.path.exists(save_file):
            continue
        print("Processing design {} ...".format(design))
        netlist_path = os.path.join(rawdata_path, '{}/implementation'.format(design))
        report_path = os.path.join(rawdata_path, '{}/report'.format(design))
        th.multiprocessing.set_sharing_strategy('file_system')

        batch_graphs = []
        graphs = []
        vfile_pairs = {}
        parser = DcParser(top_module,adder_keywords=['add_x','alu_DP_OP','div_DP_OP'],sub_keywords=['sub_x'])
        for v in os.listdir(netlist_path):
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
            hier_report = os.path.join(report_path, hier_vf[:-1] + 'rpt')
            hier_vf = os.path.join(netlist_path, hier_vf)
            vf = os.path.join(netlist_path, vf)
            print("Processing file {}".format(vfile_pair[1]))
            # parse single file
            g, buff_replace = parser.parse((hier_vf, vf), hier_report)
            aig_nodes, aig_edges = graph2aig(g,buff_replace)
            dgl_g = aig2dglG(aig_nodes, aig_edges)
            print(dgl_g)
            graphs.append(dgl_g)
        batch_graph = dgl.batch(graphs)
        topo = dgl.topological_nodes_generator(batch_graph)
        with open(save_file,'wb') as f:
            pickle.dump((batch_graph,topo),f)
