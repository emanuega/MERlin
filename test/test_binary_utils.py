import random
import numpy as np

from merlin.util import binary

def test_bit_array_to_int_conversion():
    for i in range(50):
        intIn = random.getrandbits(64) 
        listOut = binary.int_to_bit_list(intIn, 64)
        intOut = binary.bit_list_to_int(listOut)
        assert intIn == intOut

def test_flip_bit():
    barcode = [random.getrandbits(1) for i in range(128)]
    barcodeCopy = np.copy(barcode)
    for i in range(len(barcode)):
        flippedBarcode = binary.flip_bit(barcode, i)
        assert np.array_equal(barcode, barcodeCopy)
        assert all([barcode[j] == flippedBarcode[j] \
                for j in range(len(barcode)) if j != i])
        assert barcode[i] == (not flippedBarcode[i])
