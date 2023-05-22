import sys

sys.path.append("..")
from generate_options import get_options
import os
import random
import pickle

# example of a truthtable
# module tb1(
# input [1:0] I,
# output reg O
#     );
#      always@(*)
#         case(I)
#             2'b00: O = 0;
#             2'b01: O = 1;
#             2'b10: O = 1;
#             2'b11: O = 0;
#         endcase
# endmodule


visited_values = {}

num_input = get_options().num_input
save_dir = get_options().save_dir

def full_array(elements,visited):
    res = []

    if len(elements)-len(visited)==1:
        remain = list(set(elements).difference(set(visited)))[0]
        return [str(remain)]
    for element in elements:

        if element in visited:
            continue
        visited.append(element)
        for sub_array in full_array(elements,visited):
            res.append(str(element)+sub_array)
        visited.remove(element)
    #print(res)
    return res


def get_equal_arrays(num_input,var_arrays):

    equal_arrays = []
    varValues = []
    for i in range(pow(2,num_input)):
        value = bin(i)[2:]
        while len(value)!=num_input:
            value = '0'+value
        varValues.append(value)
    #print(varValues)
    #var_arrays = full_array(range(num_input),[])
    for var_array in var_arrays:
        truthValue = list(range(pow(2,num_input)))
        for i,varValue in enumerate(varValues):
            new_value = ''
            for j in range(len(varValue)):
                new_value = varValue[int(var_array[j])]+new_value
            #print(new_value,int(new_value))
            truthValue[int(new_value,2)] = i
        equal_arrays.append(truthValue)

    #print(equal_values)
    return equal_arrays

def bitsint(bits):
    res = 0
    for bit in bits:
        res += pow(2,bit)
    return res
res = full_array(range(3),[])


visited = {}

# get the full array of truthvalues with 'num_input' inputs
var_arrays = full_array(range(num_input),[])
# calucalte the truthvalues that are equivalent
equal_arrays = get_equal_arrays(num_input,var_arrays)

print(len(equal_arrays))

num_sample = None
if get_options().num_input == 2:
    num_sample = 10
elif get_options().num_input == 3:
    num_sample = 50
elif get_options().num_input == 4:
    num_sample = 512
elif get_options().num_input == 5:
    num_sample = 50000
elif get_options().num_input == 6:
    num_sample = 100000
elif get_options().num_input == 7:
    num_sample = 100000
else:
    num_sample = 200000

current_num = 0
# sampled = []
save_path = os.path.join(save_dir,'i{}'.format(num_input))
if not os.path.exists(save_path):
    os.makedirs(save_path)

# check for the already used truthvalues, and skip these values
for vf in os.listdir(save_path):
    if not vf.endswith('.v') :
        continue
    current_num +=1
    value = int(vf.split('.')[0])
    visited[value]=True
    truthValue = bin(value)[2:]
    while len(truthValue) < pow(2, num_input):
        truthValue = '0' + truthValue
    # deal with symmetrical equivalences
    postive_postions = []
    for j in range(len(truthValue)):
        if truthValue[j] == '1':
            postive_postions.append(j)

    for array in equal_arrays:
        equal_value = 0
        for position in postive_postions:
            equal_value += pow(2, pow(2, num_input) - 1 - array[position])
        visited[equal_value] = True
    # deal with complementary
    visited[pow(2, pow(2, num_input)) - 1 - value] = True


print('num visited:',len(visited),'total',pow(2,pow(2,num_input)))
print('equal transformation: ',len(equal_arrays))
size =0
# Iteratively generate truthtable-circuits with random truthvalues.
# We use a map 'visited' to save all the used truthvalues and their complimentaries,
# so that no duplicate truthtable is generated.
while current_num<num_sample:
    #i = random.randint(1,pow(2,pow(2,num_input))-1)
    num = ''
    # generate a random binary list 'num' that represents the truthvalue
    # we use positive_positions to save the position of bits that has value 1
    postive_postions = []
    for j in range(pow(2,num_input)):
        bit = random.randint(0,1)
        if bit==1:
            postive_postions.append(j)
        num += str(bit)
    # tranfer the binary list 'num' to a int number i
    i = int(num,2)
    # check if i is used or not
    if visited.get(i,False):
        continue
    size += 1
    print(size,i)
    current_num += 1
    visited[i] = True

    # set the equivalent truthvalues as visited
    for array in equal_arrays:
        equal_value = 0
        for position in postive_postions:
            equal_value += pow(2,pow(2,num_input)-1-array[position])
        visited[equal_value] = True
    # deal with complementary
    visited[pow(2,pow(2,num_input))-1-i] = True

    #print(truthValue,len(truthValue))
    # generate the truthtable-circuit and save the file
    with open(os.path.join(save_path,'{}.v'.format(i)),'w') as f:
        f.write('module i{}_v{}(\n'.format(num_input,i))
        f.write('input [{}:0] I,\n'.format(num_input-1))
        f.write('output reg O\n')
        f.write(');\n')
        f.write('always@(*)\n\tcase(I)\n')
        for j in range(pow(2,num_input)):
            f.write("\t\t{}'b{}: O = {};\n".format(num_input,bin(j)[2:],num[j]))
        f.write('\tendcase\n')
        f.write('endmodule\n')

print('num visited:',len(visited),'total',pow(2,pow(2,num_input)))

# run the dc to synthesize the generated truthtables
os.system('source /opt2/synopsys/setup.sh')
options = get_options()
data_dir = os.path.join(options.save_dir,'i{}'.format(options.num_input))
tcl_file ='dc.tcl'
os.makedirs(os.path.join(data_dir,'implementation'),exist_ok=True)
for vf in os.listdir(data_dir):
    value = vf.split('.')[0]
    # skip the aready sythesized truthtables
    if os.path.exists(os.path.join(data_dir,'implementation/truthtable_i{}_v{}_d5.00.v').format(options.num_input,value)):
        #print('continue')
        continue
    if not vf.endswith('.v') :
        continue

    # modify the dc script
    with open(tcl_file,'r') as f:
        lines = f.readlines()
    lines[2] = "set numInput \"{}\"\n".format(options.num_input)
    lines[3] = "set value \"{}\"\n".format(value)

    with open(tcl_file,'w') as f:
        f.writelines(lines)
    # run dc
    os.system('dc_shell-xg-t -f {}'.format(tcl_file))