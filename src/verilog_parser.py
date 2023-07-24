import os
import collections

from typing import List, Dict, Tuple, Optional
import pyverilog
from pyverilog.vparser.parser import parse
import re

import networkx as nx

import cProfile

port_map = {
    "HADD": {"A0":"A1","B0":"A2"},
    "FADD": {"A":"A1","B":"A2","CI":"A3"}
}

def parse_arg(arg,port_info,ios,wires):
    r"""

    parse the information of an arg

    :param arg:
        the arg
    :param port_info: PortInfo
        the port that the arg belongs to
    :param ios:
        io information of the current top module
    :param wires:
        wire information of current top module
    :return:

    """

    # identifier, e.g., a
    if type(arg) == pyverilog.vparser.ast.Identifier:
        if wires.get(arg.name,None) is not None:
            high_bit, low_bit = wires[arg.name]
        # if the arg is an io of the current top module, then it need chain update latter
        elif ios.get(arg.name,None) is not None:
            high_bit, low_bit = ios[arg.name]
            port_info.flag_update = True
            port_info.args_need_update.add(arg.name)
        else:
            assert False

        # add the current arg to the port's arg_list
        width = high_bit-low_bit+1
        if width == 1:
            port_info.arg_list.append(arg.name)
        else:
            for i in range(high_bit,low_bit-1,-1):
                port_info.arg_list.append("{}_{}".format(arg,i))
    # const, e.g., 1'b0
    elif type(arg) == pyverilog.vparser.ast.IntConst:
        port_info.arg_list.append(arg.value)
    # parselect, e.g., a[n1:n2]
    elif type(arg) == pyverilog.vparser.ast.Partselect:
        arg_nm,high_bit,low_bit = arg.children()
        arg_nm = arg_nm.name
        # get the highest/lowest bit
        high_bit, low_bit = int(str(high_bit)),int(str(low_bit))
        if high_bit < low_bit:
            temp = high_bit
            high_bit = low_bit
            low_bit = temp
        # add the arg to arglist
        for i in range(high_bit,low_bit-1,-1):
            port_info.arg_list.append("{}_{}".format(arg_nm,i))
        if ios.get(arg_nm,None) is not None:
            port_info.flag_update = True
            port_info.args_need_update.add(arg_nm)
    # pointer, e.g., a[n]
    elif type(arg) == pyverilog.vparser.ast.Pointer:
        arg_nm, position = arg.children()
        arg_nm = arg_nm.name
        port_info.arg_list.append("{}_{}".format(arg_nm,position))
        if ios.get(arg_nm,None) is not None:
            port_info.flag_update = True
            port_info.args_need_update.add(arg_nm)
    else:
        print(arg)
        assert False

class ModuleInfo:

    cell_name:str   #
    cell_type:str
    instance_name:str
    ports:dict
    index:int
    def __init__(self,cell_name,cell_type,instance_name):
        self.cell_name = cell_name
        self.cell_type = cell_type
        self.instance_name = instance_name
        self.ports = {}
        self.index = -1

class PortInfo:
    ptype:str
    portname: str
    argname: str
    argcomp: str
    is_adder_input: bool
    is_adder_output: bool
    is_sub_input1: bool
    is_sub_input2: bool
    is_muldiv_output:bool
    is_muldiv_input1: bool
    is_muldiv_input2: bool
    is_sub_output: bool
    input_comp: str
    output_comp: str
    arg_list: list
    position:tuple
    flag_update:bool
    args_need_update:set
    flag_mult:bool
    def __init__(self, portname,argname, argcomp):
        self.ptype = None
        self.portname = portname
        self.argname = argname
        self.argcomp = argcomp
        self.is_adder_input = False
        self.is_adder_output = False
        self.is_sub_input1 = False
        self.is_sub_input2 = False
        self.is_muldiv_output =  False
        self.is_muldiv_input1 = False
        self.is_muldiv_input2 = False
        self.is_sub_output = False
        self.arg_list = []
        self.position = None
        self.flag_update = False
        self.args_need_update = set()
        self.flag_mult = False
class DcParser:
    def __init__(
        self, top_module: str,adder_keywords: List[str], sub_keywords: List[str]
    ):
        self.top_module = top_module
        self.adder_keywords = adder_keywords
        self.sub_keywords = sub_keywords
        self.muldivs = []

    def is_input_port(self, port: str) -> bool:
        return not self.is_output_port(port)

    def is_output_port(self, port: str) -> bool:
        return port in ("Y", "S", "SO", "CO", "C1", "Q", "QN","O1")

    def parse_report(self,fname):
        r"""

        parse the sythesis report to find information about the target arithmetic blocks (cells)

        here gives the information of an example block in the report:
            ****************************************
            Design : MulAddRecFNToRaw_preMul
            ****************************************

            ......

            Datapath Report for DP_OP_279J57_124_314
            ==============================================================================
            | Cell                 | Contained Operations                                |
            ==============================================================================
            | DP_OP_279J57_124_314 | mult_292515 add_292517                              |
            ==============================================================================

            ==============================================================================
            |       |      | Data     |       |                                          |
            | Var   | Type | Class    | Width | Expression                               |
            ==============================================================================
            | I1    | PI   | Signed   | 9     |                                          |
            | I2    | PI   | Signed   | 65    |                                          |
            | I3    | PI   | Signed   | 65    |                                          |
            | T7    | IFO  | Signed   | 73    | I1 * I2                                  |
            | O1    | PO   | Signed   | 73    | T7 + I3                                  |
            ==============================================================================

            Implementation Report
            ......

        :param fname: str
            the path of the report file
        :return:
            dp_target_blocks:
                {block_name:(block_type,{input_port:position},{output_port:position})}
        """
        with open(fname,'r') as f:
            text = f.read()
        blocks  = text.split('Datapath Report for')
        blocks = blocks[1:]
        dp_target_blocks = {}

        for block in blocks:
            block = block.split('Implementation Report')[0]
            block = block[:block.rfind('\n==============================================================================')]
            block_name = block.split('\n')[0].replace(' ','')
            vars = block[block.rfind('=============================================================================='):]
            vars = vars.split('\n')[1:] # the vars in the datapath report, e.g., | I1    | PI   | Signed   | 9     |                                          |

            var_types = {}   # record the port type of the vars, valid types include: PI, PO, IFO
            for var in vars:
                var = var.replace(' ','')
                _,var_name,type,data_class,width,expression,_ =var.split('|')
                var_types[var_name] = (type)
                # find a mutiply operation
                if '*' in expression :
                    self.muldivs.append(block_name)
                    dp_target_blocks[block_name] = dp_target_blocks.get(block_name, ( 'muldiv',{}, {}) )
                    # get the operants (inputs)
                    operants = expression.split('*')
                    for operant in operants:
                        dp_target_blocks[block_name][1][operant] = 2
                # find an add operation
                if '+' in expression and '-' not in expression:
                    dp_target_blocks[block_name] = dp_target_blocks.get(block_name, ('add', {}, {}))
                    # record the output port of the block
                    dp_target_blocks[block_name][2][var_name] = 1
                    # get the operants
                    operants = expression.split('+')
                    for operant in operants:
                        dp_target_blocks[block_name][1][operant] = 1
                if 'addsub' in expression:
                    dp_target_blocks[block_name] = dp_target_blocks.get(block_name, ('add', {}, {}))
                    # record the output port of the block
                    dp_target_blocks[block_name][2][var_name] = 1
                    # get the operants
                    expression = expression[expression.find('('):expression.rfind(')')]
                    operants = expression.split(',')
                    for operant in operants:
                        dp_target_blocks[block_name][1][operant] = 1
                # find a subtract operation
                if '-' in expression and '+' not in expression:
                    dp_target_blocks[block_name] = dp_target_blocks.get(block_name, ('sub', {}, {}))
                    # record the output port of the block
                    dp_target_blocks[block_name][2][var_name] = 1
                    # get the operants
                    operants = expression.split('-')
                    for i,operant in enumerate(operants):
                        dp_target_blocks[block_name][1][operant] = 1 if i==0 else 2
        print('dp_target_blocks',dp_target_blocks)

        return dp_target_blocks

    def parse_port_hier(
        self, ios:dict,wires:dict, port: pyverilog.vparser.parser.Portlist,
    ) -> PortInfo:
        r"""

        parse a given port

        :param ios: dict
            io information of the modules
        :param wires: dict
            wires information
        :param port: pyverilog.vparser.parser.Portlist
            port
        :return:
            portInfo: PortInfo
                the information of the port
        """

        portname, argname = port.portname, port.argname
        port_info = PortInfo(portname,None,None)

        # if the arg is concated (in the form of {a,b})
        if type(argname) == pyverilog.vparser.ast.Concat:
            args = argname.children()
            for arg in args:
                parse_arg(arg,port_info,ios,wires)
        else:
            parse_arg(argname,port_info,ios,wires)

        return port_info

    def parse_port(
        self, mcomp: str,target_cells: list,port: pyverilog.vparser.parser.Portlist,index01:list,dp_inputs:list,dp_outputs:list
    ) -> PortInfo:
        r"""

        :param mcomp: str
            the module of the port
        :param target_cells: list
            target blocks information
        :param port: port
        :param index01: list
            the index of 1'b0 and 1'b1
        :param dp_inputs: list
        :param dp_outputs: list
        :return:
            port_info: PortInfo
                the information of the port
        """
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
        position = None

        if argname == "1'b0" :
            argname = "{}_{}".format(argname,index01[0])
            index01[0] += 1
        elif argname =="1'b1":
            argname = "{}_{}".format(argname, index01[1])
            index01[1] += 1

        port_info = PortInfo(portname, argname, argcomp)

        if portname in ("CLK"):  # clock
            port_info.ptype = "CLK"
            return port_info
        elif self.is_output_port(portname):
            port_info.ptype = "fanout"
        else:
            port_info.ptype = "fanin"

        for muldiv in self.muldivs:
            if muldiv in mcomp:
                port_info.flag_mult = True
                break
        is_target = False
        for kw in self.adder_keywords:
            if kw in mcomp :
                is_target = True
                break
        for kw in self.sub_keywords:
            if kw in mcomp :
                is_target = True
                break
        if len(dp_inputs)!=0 or len(dp_outputs)!=0:
            is_target = True

        # label the io of target blocks
        if is_target and mcomp != argcomp:
            module_ports = None
            # for cases that instance_name is not unique, e.g, have several add_x_1，each is instance of different cell,
            # in theses cases, mcomp contains both cell information and instance information
            cell_type = None
            for module_info in target_cells:
                if module_info.instance_name.lower() in mcomp.lower():
                    module_ports = module_info.ports
                    cell_type = module_info.cell_type
                    break
            if module_ports is None:
                print('module_ports is none', mcomp, portname, argname)
                return port_info

            # get the position of the arg
            for mport in module_ports.keys():
                mport_args = module_ports[mport]
                for i, arg in enumerate(mport_args):
                    if arg.lower() in argname.lower():
                        position = (mport, len(mport_args) - 1 - i)
                        break

            if position is None:
                if "1'b0" in argname or "1'b1" in argname:
                    return port_info
                if re.match("n\d+$", argname) is not None:

                    return port_info
                #print(mcomp)
                #print(portname, argname)
                # print(module_ports)
                # position = False
                pos = argname.split('_')[-1]
                if re.match('\d+$',pos) is None:
                    position = ('E',0)
                else:
                    position = ('E', int(pos))

            port_info.position = position
            # label the ouput wires
            if self.is_output_port(portname) :
                if len(dp_outputs) != 0 and position[0] not in dp_outputs.keys():
                    return port_info

                if cell_type == 'add':
                    port_info.is_adder_output = True
                elif cell_type == 'sub':
                    port_info.is_sub_output = True
                elif cell_type == 'muldiv':
                    port_info.is_muldiv_output = True
                else:
                    print(cell_type)
                    assert  False

                port_info.output_comp = mcomp
            # label the input wires
            else:
                if len(dp_inputs) != 0 and position[0] not in dp_inputs.keys():
                    return port_info
                if cell_type == 'add':
                    port_info.is_adder_input = True

                elif cell_type == 'sub':
                    if len(dp_inputs)!=0:

                        sub_position = dp_inputs[position[0]]
                        if sub_position == 1:
                            port_info.is_sub_input1 = True
                        else:
                            port_info.is_sub_input2 = True

                    else:
                        if position[0] == 'A' :
                            port_info.is_sub_input1 = True
                        elif position[0] == 'B' :
                            port_info.is_sub_input2 = True
                        else:
                            print(mcomp,position[0],port_info.portname)
                            return port_info
                elif cell_type == 'muldiv':
                    if len(dp_inputs)!=0:
                        sub_position = dp_inputs[position[0]]
                        if sub_position == 1:
                            port_info.is_muldiv_input1 = True
                        else:
                            port_info.is_muldiv_input2 = True

                    else:
                        if position[0] in ('I1','I2') :
                            port_info.is_muldiv_input2 = True
                        elif position[0] == 'I3' :
                            port_info.is_muldiv_input1 = True
                        else:
                            print(mcomp,position[0],port_info.portname)
                            return port_info
                else:
                    print(cell_type)
                    assert False
                port_info.input_comp = mcomp

        elif is_target and argcomp != mcomp:
            assert False

        return port_info

    def parse_hier(self, fname,dp_target_blocks):
        r"""

        parse the hierarchical netlist

        :param fname: str
            netlist filepath
        :param dp_target_blocks: {block}
            a dictionary of the block information extracted by pase_report
        :return:
            target_blocks : dict
                the information of the target  arithmetic blocks
        """
        target_blocks = {}
        ast, directives = parse([fname])
        args_to_update = {}
        
        # parse the modules one by one
        for module in ast.description.definitions:
            
            ios = {}
            wires = {}
            # parse the ios and wires of the current top module 
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
                        # save the highest/lowest bit of each io / wire
                        if type(decl) == pyverilog.vparser.ast.Input or type(decl) == pyverilog.vparser.ast.Output:
                            ios[name] = (high_bit, low_bit)
                        else:
                            wires[name] = (high_bit, low_bit)
                elif type(sentence) == pyverilog.vparser.ast.Wire:
                    name = sentence.name
                    wires[name] = (0,0)
            
            # parse each module/cell in the current top module
            for item in module.items:
                if type(item) != pyverilog.vparser.ast.InstanceList:
                    continue
                instance = item.instances[0]
                # we extract the following parts:
                # mcell: cell name in SAED, e.g. AND2X1
                # mname: module name, e.g. ALU_DP_OP_J23_U233
                mcell = instance.module 
                mname = instance.name
                mcomp = mname[:mname.rfind('_')]
                ports = instance.portlist

                if mcell.startswith("SNPS_CLOCK") or mcell.startswith("PlusArgTimeout"):
                    continue

                # judge if the current module a target one (target arithemetic block) or not
                is_target =  False
                # find if the module'name contain specific keywords, if true, then it is a target module
                for key_word in self.adder_keywords:
                    if key_word in mcomp:
                        # instance.show()
                        # print('######################', mcell, mcomp)
                        cell_type = 'add'
                        is_target = True
                        break
                for key_word in self.sub_keywords:
                    if key_word in mcomp:
                        cell_type = 'sub'
                        is_target = True
                        break

                # find if the current module in the target_blocks list extracted from the report file,
                # if true, then it is a target module
                if dp_target_blocks.get(mname,None) is not None:
                    cell_type = dp_target_blocks[mname][0]
                    is_target = True


                # parse the information of a target module
                if is_target:
                    module_info = ModuleInfo(mcell, cell_type.lower(), mname.lower())
                    for word in mcell.split('_')[:-1]:
                        if re.match('\d+$', word) is not None:
                            module_info.index = int(word)
                            break
                    # parse the port information of the module
                    for p in ports:
                        port_info = self.parse_port_hier(ios, wires, p)
                        # if some arg of the cell's port is input/output of the father module, then when the father module is instanced latter,
                        # these args should be replaced with args of corresponding port of the father module instance
                        # eg, in the following example, i1 should be replaced with w1 for cell add_x_1
                        # eg, module ALU
                        #       input [63:0] i1,
                        #       ...
                        #       CSR_inc add_x_1 (.A(i1),...)
                        #       ...
                        #     endmodule
                        #     module Rocket
                        #       ...
                        #       ALU alu (.i1(w1),...)
                        # we mantain the information of args that need to update in 'args_to_update':
                        #               {father_module_name:{(cell_type,cell_name,portname):[args need to update]} }
                        #   eg, {'ALU':{(CSR_inc,add_x_1,'A'):[i1]}}

                        if port_info.flag_update:
                            args_to_update[module.name] = args_to_update.get(module.name, {})
                            port2update = (mcell, mname.lower(), port_info.portname)
                            args_to_update[module.name][port2update] = args_to_update[module.name].get(port2update, [])
                            for arg in port_info.args_need_update:
                                args_to_update[module.name][port2update].append(arg)
                        module_info.ports[port_info.portname] = port_info.arg_list
                    
                    # record the informaion of target blocks in the current top module
                    target_blocks[module.name] = target_blocks.get(module.name, [])
                    target_blocks[module.name].append(module_info)

                # if there are some target blocks in the current top module
                if target_blocks.get(mcell,None) is not None:
                    # if some args of the target blocks need chain update latter, then we record them
                    if args_to_update.get(mcell, None) is not None:
                        ports2update = args_to_update[mcell]
                        father_ports_info = {}
                        for p in ports:
                            father_ports_info[p.portname] = self.parse_port_hier(ios, wires, p)

                        for (child_cell_name, child_instance_name,
                             child_portname), child_args2update in ports2update.items():
                            # find the portargs (arglist2update) of the child cell that need to update :
                            # eg, child_cell_info = (cell_type='CSR_inc',instance_name='add_x_1', ports={'A':[i1_63,i1_62...i1_0],'S':[...]})
                            #     arglist2update = child_cell_info.ports['A'] = [arg1_63,arg1_62...arg1_0]

                            for cell_info in target_blocks[mcell]:
                                if cell_info.cell_name == child_cell_name and child_instance_name in cell_info.instance_name :
                                    arglist2update = cell_info.ports[child_portname]

                                    # for every arg of args2update that needs to update, replace it with new arg
                                    for argname in child_args2update:
                                        #print("------ arg to update:",argname)
                                        replace_port_info = father_ports_info[argname]
                                        replace_arg_list = replace_port_info.arg_list
                                        new_args = []
                                        # print('replace portname',replace_port_info.portname)
                                        # print('replace arg list',replace_arg_list)
                                        if replace_port_info.flag_update:
                                            args_to_update[module.name] = args_to_update.get(module.name,{})
                                            port2update = (child_cell_name, child_instance_name, child_portname)
                                            args_to_update[module.name][port2update] = args_to_update[module.name].get(
                                                port2update, [])
                                            for arg in replace_port_info.args_need_update:
                                                args_to_update[module.name][port2update].append(arg)
                                        # replace the args of the child port with new args of the corresponding father port
                                        # print(arglist2update)
                                        for arg in arglist2update:
                                            if replace_port_info.portname in arg:
                                                index = arg.split('_')[-1]
                                                if re.match('\d+$', index) is not None:

                                                    new_args.append(
                                                        replace_arg_list[len(replace_arg_list) - 1 - int(index)])
                                                else:
                                                    new_args.append(replace_arg_list[0])
                                            else:
                                                new_args.append(arg)
                                        cell_info.ports[child_portname] = new_args
                                        arglist2update = new_args
                                    #     print('new arglist:',cell_info.ports[child_portname])
                                    # print('#############################################')
                                    # print(cell_info.cell_type,cell_info.instance_name,cell_info.ports)

                        args_to_update[mcell] = None

                    for module_info in target_blocks[mcell]:
                        module_info.instance_name = "{}_{}".format(mname,module_info.instance_name)
                        target_blocks[module.name] = target_blocks.get(module.name, [])
                        target_blocks[module.name].append(module_info)
                    target_blocks[mcell] = None
                # if we encounter a father module instance as above mentioned, eg, ALU alu (.i1(w1),...)
                #   we first parse the ports of the father module instance,
                #   then we find the corresponding relationship between args of father instance and args of target_child cell ,and replace


            #print(module.name,args_to_update)
        # for module,cells in target_blocks.items():
        #     print(module)
        #     if cells is not None:
        #         for cell in cells:
        #             print(cell.cell_name,cell.instance_name,cell.ports)

        target_blocks = target_blocks[self.top_module]
        # for cell in target_blocks:
        #     print(cell.cell_name, cell.instance_name, cell.ports)

        return target_blocks

    def parse_nonhier(self, fname,dp_target_blocks,target_blocks):
        r"""
        
        parse the non-hierarchical netlist with block information extracted from report and hier_netlist
        
        :param fname: str
            nonhier netlist filepath
        :param dp_target_blocks: dict
            the information of some target arithmetic blocks extraced from the hier_netlist
        :param target_blocks:  dict
            the information of all the target arithmetic blocks extraced from the report

        :return:
            nodes: list
                the labeled nodes of the transformed DAG
            edges:  list
                the edges of the transformed DAG
        """

        nodes: List[Tuple[str, Dict[str, str]]] = [ ]  # a list of (node, {"type": type})
        edges: List[
            Tuple[str, str, Dict[str, bool]]
        ] = []  # a list of (src, dst, {"is_reverted": is_reverted})

        ast, directives = parse([fname])
        index01 = [0,0]
        adder_inputs = set()
        adder_outputs = set()
        sub_inputs1 = set()
        sub_inputs2 = set()
        sub_outputs = set()
        multdiv = set()
        muldiv_inputs1 = set()
        muldiv_inputs2 = set()
        multdiv_outputs = set()
        buff_replace = {}
        top_module = None

        positions = {}

        for module in ast.description.definitions:
            if module.name == self.top_module:
                top_module = module
                break
        assert top_module is not None, "top module {} not found".format(self.top_module)
        print(len(top_module.items))
        # parse the information of each cell/block
        for item in top_module.items:
            if type(item) != pyverilog.vparser.ast.InstanceList:
                continue
            instance = item.instances[0]

            # we extract the following parts:
            # mcell: cell name in SAED, e.g. AND2X1
            # mtype: cell type with input shape, e.g. AND2
            # mfunc: cell function, e.g. AND
            # mname: module name, e.g. ALU_DP_OP_J23_U233
            # mcomp: module component, e.g. ALU_DP_OP_J23
            mcell = instance.module  # e.g. AND2X1
            mname = instance.name
            ports = instance.portlist
            mtype = mcell[0 : mcell.rfind("X")]  # e.g. AND2
            mfunc = mtype  # e.g. AND
            mcomp = mname[: mname.rfind("_")]
            if mcell.startswith("SNPS_CLOCK") or mcell.startswith("PlusArgTimeout"):
                continue

            # fanins / fanouts the the cell
            fanins: List[PortInfo] = []
            fanouts: List[PortInfo] = []

            dp_inputs,dp_outputs = [],[]

            # judge if the current cell a target
            for dp_cell in dp_target_blocks.keys():
                if dp_cell in mcomp:
                    dp_inputs = dp_target_blocks[dp_cell][1]
                    dp_outputs = dp_target_blocks[dp_cell][2]
                    break

            # parse the port information
            for p in ports:
                port_info = self.parse_port(mcomp, target_blocks,p,index01,dp_inputs,dp_outputs)
                if port_info.ptype == "fanin":
                    fanins.append(port_info)
                elif port_info.ptype == "fanout":
                    fanouts.append(port_info)

                if port_info.is_adder_input:
                    adder_inputs.add(port_info.argname)
                if port_info.is_adder_output:
                    adder_outputs.add(port_info.argname)
                if port_info.is_muldiv_output:
                    multdiv_outputs.add(port_info.argname)
                if port_info.is_muldiv_input1:
                    muldiv_inputs1.add(port_info.argname)
                if port_info.is_muldiv_input2:
                    muldiv_inputs2.add(port_info.argname)
                if port_info.is_sub_input1:
                    sub_inputs1.add(port_info.argname)
                if port_info.is_sub_input2:
                    sub_inputs2.add(port_info.argname)
                if port_info.is_sub_output:
                    sub_outputs.add(port_info.argname)
                if port_info.flag_mult:
                    multdiv.add(port_info.argname)
                if positions.get(port_info.argname,None) is None:
                    positions[port_info.argname] = port_info.position
            if not fanouts:
                item.show()
                print("***** warning, the above gate has no fanout recognized! *****")
                # do not assert, because some gates indeed have no fanout...
                # assert False, "no fanout recognized"
            inputs = {}

            for fo in fanouts:
                # the nodes are the fanouts of cells
                # do some replacement, replace some of the cell to some fix cell type, e.g., AO221 -> AND + OR
                if mfunc == "HADD":
                    if fo.portname == "SO":
                        ntype = 'XOR2'
                    elif fo.portname == "C1":
                        ntype = 'AND2'
                    else:
                        print(fo.portname)
                        assert False
                elif mfunc == "FADD":
                    if fo.portname == "S":
                        ntype = 'XOR3'
                    elif fo.portname == "CO":
                        ntype = 'MAJ'
                    else:
                        print(fo.portname)
                        assert False
                elif mfunc in ["1'b0","1'b1"] or 'DFF' in mfunc:
                    ntype = 'PI'
                else:
                    ntype = mfunc

                if 'AO' in ntype or 'OA' in ntype:

                    num_inputs = ntype[re.search('\d',ntype).start():]

                    ntype1 = 'AND2' if 'AO' in ntype else 'OR2'
                    ntype2 = 'OR{}'.format(len(num_inputs)) if 'AO' in ntype else 'AND{}'.format(len(num_inputs))
                    if 'I' in ntype:
                        output_name = '{}_i'.format(fo.argname)
                        inputs[fo.argname] = inputs.get(fo.argname, {})
                        inputs[fo.argname]['A'] = output_name
                        nodes.append((fo.argname, {"type": 'INV',"inputs":inputs[fo.argname]}))
                    else:
                        output_name = fo.argname
                    inputs[output_name] = inputs.get(output_name,{})
                    for i,num_input in enumerate(num_inputs):
                        if num_input == '2':
                            h_node_name = '{}_h{}'.format(fo.argname,i)
                            inputs[h_node_name] = inputs.get(h_node_name,{})
                            inputs[h_node_name]['A1'] = fanins[2*i].argname
                            inputs[h_node_name]['A2'] = fanins[2*i+1].argname
                            nodes.append((h_node_name, {"type": ntype1, "inputs":inputs[h_node_name]}))
                            inputs[output_name]['A{}'.format(i+1)] = h_node_name

                        elif num_input =='1':
                            inputs[output_name]['A{}'.format(i+1)] = fanins[2*i].argname
                        else:
                            print(ntype,i,num_input)
                            assert  False
                    nodes.append((output_name, {"type": ntype2, "inputs":inputs[output_name]}))
                else:
                    pos = re.search("\d", mtype)
                    if pos:
                        ntype = ntype[: pos.start()+1]

                    inputs[fo.argname] = inputs.get(fo.argname,{})
                    for fi in fanins:

                        if 'DELLN' in ntype or 'NBUFF' in ntype:
                            #print(ntype, fo.argname, fi.argname)
                            buff_replace[fo.argname] = fi.argname
                        else:
                            if port_map.get(mfunc,None) is not None:
                                port_name = port_map[mfunc][fi.portname]
                            else:
                                port_name = fi.portname
                            inputs[fo.argname][port_name] = fi.argname

                    if ntype == 'IBUFF':
                        ntype = 'INV'
                    if buff_replace.get(fo.argname,None) is None:
                        nodes.append((fo.argname, {"type": ntype,"inputs":inputs[fo.argname]}))
            # the edges represents the connection between the fanins and the fanouts
            if 'DFF' not in mfunc:
                for output,input in inputs.items():
                    for fi in input.values():
                        edges.append(
                            (
                                fi,
                                output,
                                {"is_reverted": False, "is_sequencial": "DFF" in mtype},
                            )
                        )

        # remove the buffers

        new_edges = []
        for edge in edges:
            if buff_replace.get(edge[0],None) is not None:
                new_edges.append((buff_replace[edge[0]],edge[1],edge[2]) )
            else:
                new_edges.append(edge)
        edges = new_edges
        print(
            "#inputs:{}, #outputs:{}".format(len(adder_inputs), len(adder_outputs)),
            flush=True,
        )


        # add the edges that connect PIs
        gate_names = set([n[0] for n in nodes])
        pis = []
        for (src, _, _) in edges:
            if src not in gate_names and src not in pis:
                nodes.append((src, {"type": "PI", "inputs":{}}))
                pis.append(src)

        # label the nodes
        for n in nodes:
            n[1]["is_adder_input"] = n[0] in adder_inputs
            n[1]["is_adder_output"] = n[0] in adder_outputs
            n[1]["position"] = positions.get(n[0], None)
            if n[0] in multdiv:
                n[1]['is_adder_input'] = -1
                n[1]['is_adder_output'] = -1
            n[1]['is_mul_output'] = n[0] in multdiv_outputs
            if n[0] in muldiv_inputs1:
                n[1]['is_mul_input'] = 1
            elif n[0] in muldiv_inputs2:
                n[1]['is_mul_input'] = 2
            else:
                n[1]['is_mul_input'] = 0

            n[1]['is_sub_output'] = n[0] in sub_outputs
            if n[0] in sub_inputs1:
                n[1]['is_sub_input'] = 1
            elif n[0] in sub_inputs2:
                n[1]['is_sub_input'] = 2
            else:
                n[1]['is_sub_input'] = 0

        print('num adder inputs:', len(adder_inputs))
        print('num adder outputs:', len(adder_outputs))

        print('num muldiv inputs1:', len(muldiv_inputs1))
        print('num muldiv inputs2:', len(muldiv_inputs2))
        print('num muldiv outputs:', len(multdiv_outputs))

        print('num sub inputs1:', len(sub_inputs1))
        print('num sub inputs2:', len(sub_inputs2))
        print('num sub outputs:', len(sub_outputs))

        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        # g_topo = nx.topological_sort(g)
        # for n in g_topo:
        #     print(n,g.nodes[n]['type'], g.nodes[n]['is_adder_output'], g.nodes[n]['inputs'])
        return g, buff_replace

    def parse(self,vfile_pair,hier_report):
        R"""

        :param vfile_pair: (str, str)
            The netlist files of the target circuit, containing a hierarchical one where the hierarchy (boundary of modules) is preserved
            and a non-hierarchical one where the hirerarchy is cancelled
        :param hier_report: str
            the report file given by DC
        :return: (nodes:list, edges:list)
            return the nodes and edges of the transformed DAG
        """

        hier_vf, nonhier_vf = vfile_pair[0], vfile_pair[1]
        dp_target_blocks = self.parse_report(hier_report)
        target_cells = self.parse_hier(hier_vf, dp_target_blocks)
        graph = self.parse_nonhier(nonhier_vf, dp_target_blocks=dp_target_blocks,target_blocks=target_cells)

        return graph

