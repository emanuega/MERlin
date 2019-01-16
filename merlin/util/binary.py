import numpy as np
from typing import List


def bit_list_to_int(bitList: List[bool]) -> int:
    """Converts a binary list to an integer

    Args:
        bitList: the binary list to convert
    Returns:
        The integer corresponding to the input bit list
    """
    out = 0
    for b in reversed(bitList):
        out = (out << 1) | b
    return out

def int_to_bit_list(intIn: int, bitCount: int) -> List[bool]:
    """Converts an integer to a binary list with the specified number of bits.

    Args:
        intIn: the integer to convert
        bitCount: the number of bits to include in the output bit list
    Returns:
        A list of bit that specifies the input integer. The least significant
            bit is first in the list.
    """
    return [k_bit_set(intIn, k) for k in range(bitCount)]

def k_bit_set(n: int, k: int) -> bool:
    """Determine if the k'th bit of integer n is set to 1.

    Args:
        n: the integer to check
        k: the index of the bit to check where 0 corresponds with the least
            significant bit
    Returns:
         true if the k'th bit of the integer n is 1, otherwise false. If
            k is None, this function returns None.
    """
    if k is None:
        return None

    if n & (1 << k):
        return True
    else:
        return False

def flip_bit(barcode: List[bool], bitIndex: int) -> List[bool]:
    """Generates a version of the provided barcode where the bit at the
    specified index is inverted.

    The provided barcode is left unchanged. It is copied before flipping the
    bit.

    Args:
        barcode: A binary array where the i'th entry corresponds with the 
            value of the i'th bit
        bitIndex: The index of the bit to reverse
    Returns:
        A copy of barcode with bitIndex inverted
    """
    bcCopy = np.copy(barcode)
    bcCopy[bitIndex] = not bcCopy[bitIndex]
    return bcCopy
