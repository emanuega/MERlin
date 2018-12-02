#TODO array should be changed to list to be consistent with python nomenclature
def bit_array_to_int(bitArray):
    '''Converts a binary array to an integer'''
    out = 0
    for b in reversed(bitArray):
        out = (out << 1) | b
    return out

def int_to_bit_array(intIn, bitCount):
    '''
    Converts an integer to a binary list with the specified number of bits.
    '''
    return [k_bit_set(intIn, k) for k in range(bitCount)]

def k_bit_set(n, k):
    '''
    Returns true if the k'th bit of the integer n is 1, otherwise false. If
    k is None, this function returns None. The index k is 0-based.
    '''
    if k is None:
        return None

    if n & (1 << k):
        return True
    else:
        return False

