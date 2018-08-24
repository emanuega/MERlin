
def bit_array_to_int(bitArray):
    '''Converts a binary array to an integer'''
    out = 0
    for b in bitArray:
        out = (out << 1) | b
    return out
