from math import log

position2group = {}

def gather_equivalent_positions(max_width):
    for num in range(pow(2,max_width)):
        bits = bin(num)[2:]
        group_id = len(bits.replace('0',''))
        position2group[num] = group_id
        #print(num,bits,group_id)

gather_equivalent_positions(7)


def truthvalue2code(value,width):
    bits = bin(value)[2:].zfill(pow(2,width))
    bits = [int(bit) for bit in bits]
    group_count = (width+1)*[0]
    group_size = (width+1)*[0]
    #print(width,bits)
    for i,bit in enumerate(bits):
        group = position2group[i]
        group_count[group] += bit
        group_size[group] += 1
    #print(group_count)

    return [c/s for (c,s) in zip(group_count,group_size)]
