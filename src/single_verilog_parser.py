import os
import collections

from typing import List, Dict, Tuple, Optional
import pyverilog
from pyverilog.vparser.parser import parse
import re

import networkx as nx

import cProfile
class PortInfo:
    ptype: str
    portname: str
    argname: str
    argcomp: str
    is_output:str

    def __init__(self, portname, argname, argcomp):
        self.ptype = None
        self.portname = portname
        self.argname = argname
        self.argcomp = argcomp
        self.is_output = False


class DcParser:
    def __init__(
            self, top_module: str
    ):
        self.top_module = top_module
        self.hadd_name_dict = {}
        self.hadd_name_dict["hadd_s"] = "XOR2"
        self.hadd_name_dict["hadd_c"] = "AND2"
        self.hadd_name_dict["fadd_s"] = "XOR3"
        self.hadd_name_dict["fadd_c"] = "MAJ"


    def is_input_port(self, port: str) -> bool:
        return not self.is_output_port(port)

    def is_output_port(self, port: str) -> bool:
        return port in ("Y", "S", "SO", "CO", "C1", "Q", "QN")

    def parse_port(
            self, port: pyverilog.vparser.parser.Portlist,ios
    ) -> PortInfo:
        portname, argname = port.portname, port.argname
        if type(argname) == pyverilog.vparser.ast.Partselect:
            print(argname)
        if type(argname) == pyverilog.vparser.ast.Pointer:
            argname = str(argname.var) + "_" + str(argname.ptr)
        elif type(argname) == pyverilog.vparser.ast.IntConst:
            argname = argname.__str__()
        else:  # just to show there could be various types!
            argname = argname.__str__()
        argcomp = argname[: argname.rfind("_")]


        port_info = PortInfo(portname, argname, argcomp)

        if portname in ("CLK"):  # clock
            port_info.ptype = "CLK"
        elif self.is_output_port(portname):
            port_info.ptype = "fanout"
        else:
            port_info.ptype = "fanin"

        for io in ios:
            #if re.match(io+'_d',argname):
            if argcomp==io:
                #print(argcomp)
                port_info.is_output = True
        return port_info


    def parse(self, fname):
        """ parse dc generated verilog """
        # adder_cells = set()

        PIs: List[str] = []  # a list of PI nodes
        POs: List[str] = []  # a list of PO nodes
        mult_infos = {}  # {mcomp:([(mult_input_wire,position)],[mult_output_wire,position])}
        nodes: List[Tuple[str, Dict[str, str]]] = [
        ]  # a list of (node, {"type": type})
        edges: List[
            Tuple[str, str, Dict[str, bool]]
        ] = []  # a list of (src, dst, {"is_reverted": is_reverted})
        num_wire = 0
        buff_replace = {}
        ast, directives = parse([fname])
        index01 = [0, 0]

        output_node = None

        top_module = None

        for module in ast.description.definitions:
            top_module = module
            ios = []
            for sentence in module.children():
                if type(sentence) == pyverilog.vparser.ast.Decl:

                    for decl in sentence.children():
                        name = decl.name
                        if decl.width is None:
                            high_bit, low_bit = 0, 0
                        else:
                            high_bit, low_bit = decl.width.children()
                            high_bit,low_bit = int(high_bit.value),int(low_bit.value)
                            if high_bit<low_bit:
                                temp = high_bit
                                high_bit = low_bit
                                low_bit = temp
                        if type(decl) == pyverilog.vparser.ast.Output:
                            # if type(decl) == pyverilog.vparser.ast.Output and re.match('io_pmp_\d_addr',decl.name):
                            #     decl.show()
                            ios.append(name)
        #print(ios)
        assert top_module is not None, "top module {} not found".format(self.top_module)

        for item in top_module.items:
            if type(item) != pyverilog.vparser.ast.InstanceList:
                continue
            instance = item.instances[0]

            # we extract the following parts:
            # mcell: cell name in SAED, e.g. AND2X1
            # mtype: cell type with input shape, e.g. AND2
            # mfunc: cell function, e.g. AND2
            # mname: module name, e.g. ALU_DP_OP_J23_U233
            # mcomp: module component, e.g. ALU_DP_OP_J23
            mcell = instance.module  # e.g. AND2X1
            mname = instance.name
            ports = instance.portlist
            mtype = mcell[0: mcell.rfind("X")]  # e.g. AND2
            mfunc = mtype  # e.g. AND2

            # pos = re.search("\d", mtype)
            # if pos:
            #     mfunc = mtype[: pos.start()]
            mcomp = mname[: mname.rfind("_")]
            if mcell.startswith("SNPS_CLOCK") or mcell.startswith("PlusArgTimeout"):
                continue
            fanins: List[PortInfo] = []
            fanouts: List[PortInfo] = []

            for p in ports:
                port_info = self.parse_port(p,ios)
                if port_info.is_output:
                    #output_node = port_info.argname
                    POs.append(port_info.argname)
                if port_info.ptype== "fanin" :
                    fanins.append(port_info)
                elif port_info.ptype == "fanout" :
                    fanouts.append(port_info)
            if not fanouts:
                item.show()
                print("***** warning, the above gate has no fanout recognized! *****")
            inputs = {}
            # print(mfunc,mname)
            for fo in fanouts:
                if mfunc == "HADD":
                    if fo.portname == "SO":
                        ntype = self.hadd_name_dict["hadd_s"]
                    elif fo.portname == "C1":
                        ntype = self.hadd_name_dict["hadd_c"]
                    else:
                        print(fo.portname)
                        assert False
                elif mfunc == "FADD":
                    if fo.portname == "S":
                        ntype = self.hadd_name_dict["fadd_s"]
                    elif fo.portname == "CO":
                        ntype = self.hadd_name_dict["fadd_c"]
                    else:
                        print(fo.portname)
                        assert False
                else:
                    ntype = mfunc

                if 'AO' in ntype or 'OA' in ntype:

                    num_inputs = ntype[re.search('\d', ntype).start():]
                    ntype1 = 'AND' if 'AO' in ntype else 'OR'
                    ntype2 = 'OR' if 'AO' in ntype else 'AND'
                    if 'I' in ntype:
                        output_name = '{}_i'.format(fo.argname)
                        nodes.append((output_name, {"type": ntype2}))
                        nodes.append((fo.argname, {"type": 'INV'}))
                        inputs[fo.argname] = [output_name]

                        # edges.append((output_name,fo.argname,
                        #               {"is_reverted": False, "is_sequencial": "DFF" in mtype}))
                    else:
                        output_name = fo.argname
                        nodes.append((output_name, {"type": ntype2}))
                    inputs[output_name] = inputs.get(output_name, [])
                    for i, num_input in enumerate(num_inputs):
                        if num_input == '2':
                            h_node_name = '{}_h{}'.format(fo.argname, i)
                            nodes.append((h_node_name, {"type": ntype1}))
                            inputs[h_node_name] = inputs.get(h_node_name, [])
                            inputs[h_node_name].append(fanins[2 * i].argname)
                            inputs[h_node_name].append(fanins[2 * i + 1].argname)

                            inputs[output_name].append(h_node_name)
                            # edges.append((fanins[2*i].argname,h_node_name,
                            #              {"is_reverted": False, "is_sequencial": "DFF" in mtype}))
                            # edges.append((fanins[2 * i+1].argname, h_node_name,
                            #              {"is_reverted": False, "is_sequencial": "DFF" in mtype}))
                            # edges.append((h_node_name,output_name,
                            #               {"is_reverted": False, "is_sequencial": "DFF" in mtype}))
                        elif num_input == '1':
                            inputs[output_name].append(fanins[2 * i].argname)
                            # edges.append((fanins[2*i].argname,output_name,
                            #              {"is_reverted": False, "is_sequencial": "DFF" in mtype}))
                        else:
                            print(ntype, i, num_input)
                            assert False
                # elif 'NOR' in ntype or 'XNOR' in ntype or 'NAND' in ntype or 'IBUFF' in ntype:
                #     ntype1= None
                #     if 'NOR' in ntype:
                #         ntype1 = 'OR'
                #     elif 'XNOR' in ntype:
                #         ntype1 = 'XOR'
                #     elif 'NAND' in ntype:
                #         ntype1 = 'AND'
                #     elif 'IBUFF' in ntype:
                #         ntype1 = 'NBUFF'
                #     h_node_name ="{}_h".format(fo.argname)
                #     nodes.append((h_node_name,{"type":ntype1}))
                #     nodes.append((fo.argname,{"type":"INV"}))
                #     inputs[fo.argname] = [h_node_name]
                #     inputs[h_node_name] = inputs.get(h_node_name,[])
                #     for fi in fanins:
                #         inputs[h_node_name].append(fi.argname)
                #     # edges.append((h_node_name,fo.argname,
                #     #               {"is_reverted": False, "is_sequencial": "DFF" in mtype}))
                # elif 'DFF' in ntype and port_info.portname=='QN':
                #     #print(ntype,port_info.argname,port_info.position)
                #     ntype1 = 'DFFN'
                #     h_node_name = "{}_h".format(fo.argname)
                #     nodes.append((h_node_name, {"type": ntype1}))
                #     nodes.append((fo.argname, {"type": "INV"}))
                #     inputs[fo.argname] = [h_node_name]
                #     inputs[h_node_name] = inputs.get(h_node_name, [])
                #     for fi in fanins:
                #         inputs[h_node_name].append(fi.argname)
                #     # edges.append((h_node_name,fo.argname,
                #     #               {"is_reverted": False, "is_sequencial": "DFF" in mtype}))

                else:

                    pos = re.search("\d", mtype)
                    if pos:
                        #print(ntype)
                        ntype = ntype[: pos.start()]
                    # if 'DFF' in ntype :
                    #     ntype = 'DFF' if port_info.portname =='Q' else 'DFFN'
                    #     if ntype == '':
                    #         exit()
                    inputs[fo.argname] = inputs.get(fo.argname, [])
                    for fi in fanins:
                        # dff ignore SET/RESET/CLOCK
                        # if 'DFF' in ntype and fi.portname!='D':
                        #     continue
                        # if ntype == 'NBUFF' or ('DFF' in ntype and fo.portname=='Q'):

                        if ntype == 'NBUFF':
                            buff_replace[fo.argname] = fi.argname
                        else:
                            inputs[fo.argname].append(fi.argname)

                    # if ntype == 'IBUFF' or ('DFF' in ntype and fo.portname=='QN'):
                    if ntype == 'IBUFF':
                        ntype = 'INV'

                    if buff_replace.get(fo.argname, None) is None:
                        nodes.append((fo.argname, {"type": ntype}))

            for output, input in inputs.items():

                for fi in input:
                    edges.append(
                        (
                            fi,
                            output,
                            {"is_reverted": False, "is_sequencial": "DFF" in mtype},
                        )
                    )
        new_edges = []
        for edge in edges:
            if buff_replace.get(edge[0], None) is not None:
                new_edges.append((buff_replace[edge[0]], edge[1], edge[2]))
            else:
                new_edges.append(edge)
        edges = new_edges
        gate_names = set([n[0] for n in nodes])
        pis = []
        for (src, _, _) in edges:
            if src not in gate_names and src not in pis:
                if "1'b0" in src :
                    nodes.append((src,{"type":"1'b0"}))
                elif "1'b1" in src :
                    nodes.append((src,{"type":"1'b1"}))
                else:
                    nodes.append((src, {"type": "PI"}))
                pis.append(src)


        for n in nodes:
            n[1]['is_output'] = n[0] in POs
        nid = 0
        new_nodes = []
        node2id = {}
        new_edges = []
        #output_nid = None
        # for n in nodes:
        #     print(n[1]['type'])

        return nodes,edges
        #return output_node,nodes, edges



def main():
    folder = "./implementation/test/"
    total_nodes = 0
    total_edges = 0
    ntype = set()

    parser = DcParser("Rocket", hier_keywords=["add", "inc"], adder_keywords=['add_x', 'alu_DP_OP', 'div_DP_OP'],
                      hadd_type="xor")
    for v in os.listdir(folder):
        if v.endswith('.v') is False:
            continue
        nodes,edges = parser.parse(os.path.join(folder,v))
        print("nodes {}, edges {}".format(len(nodes), len(edges)))
        for n in nodes:
            ntype.add(n[1]["type"])
        total_nodes += len(nodes)
        total_edges += len(edges)
        # break
        print(ntype)

    print(total_nodes, total_edges)


if __name__ == "__main__":
    # dc_parser("../dc/simple_alu/implementation/alu_d0.20_r2_bounded_fanout_adder.v")
    main()
    # cProfile.run("main()")
