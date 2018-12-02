import random

from merlin.util import binary

def test_bit_array_to_int_conversion():
    for i in range(50):
        intIn = random.getrandbits(64) 
        listOut = binary.int_to_bit_array(intIn, 64)
        intOut = binary.bit_array_to_int(listOut)
        assert intIn == intOut
