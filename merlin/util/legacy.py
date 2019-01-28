import struct
import pandas
import numpy as np
from typing import BinaryIO
from typing import Tuple
from typing import List
from typing import Dict
from typing import Iterator


"""
This module contains convenience functions for reading and writing MERFISH
analysis results created from the deprecated Matlab pipeline.
"""


def read_blist(bFile: BinaryIO) -> pandas.DataFrame:
    entryCount, _, entryFormat = _read_binary_header(bFile)
    bytesPerEntry = int(np.sum(
        [struct.calcsize(typeNames[x['type']]) * np.prod(x['size']) for x in
         entryFormat]))
    return pandas.DataFrame(
        [_parse_entry_bytes(bFile.read(bytesPerEntry), entryFormat) for i in
         range(entryCount)])


typeNames = {'int8': 'b',
             'uint8': 'B',
             'int16': 'h',
             'uint16': 'H',
             'int32': 'i',
             'uint32': 'I',
             'int64': 'q',
             'uint64': 'Q',
             'float': 'f',
             'single': 'f',
             'double': 'd',
             'char': 's'}


def _chunker(seq, size: int) -> Iterator:
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def _read_binary_header(bFile: BinaryIO) -> Tuple[int, int, List[Dict]]:
    version = struct.unpack(typeNames['uint8'], bFile.read(1))[0]
    bFile.read(1)
    entryCount = struct.unpack(typeNames['uint32'], bFile.read(4))[0]
    headerLength = struct.unpack(typeNames['uint32'], bFile.read(4))[0]
    layout = bFile.read(headerLength).decode('utf-8').split(',')
    entryList = [
        {'name': x, 'size': np.array(y.split('  ')).astype(int), 'type': z}
        for x, y, z in _chunker(layout, 3)]
    return entryCount, headerLength, entryList


def _parse_entry_bytes(byteList, entryFormat: List[Dict]):
    entryData = {}
    byteIndex = 0
    for currentEntry in entryFormat:
        itemCount = int(np.prod(currentEntry['size']))
        itemType = typeNames[currentEntry['type']]
        itemSize = struct.calcsize(itemType)
        items = np.array([struct.unpack(
            itemType, byteList[byteIndex
                               + i * itemSize:byteIndex
                               + (i + 1) * itemSize])[0]
                          for i in range(itemCount)])
        byteIndex += itemSize * itemCount

        if currentEntry['size'][0] == 1 and currentEntry['size'][1] == 1:
            items = items[0]
        if currentEntry['size'][0] != 1 and currentEntry['size'][1] != 1:
            items = items.reshape(currentEntry['size'])

        entryData[currentEntry['name']] = items

    return entryData
