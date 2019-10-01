import numpy as np
import os

import merlin
from merlin.util import imagereader
from merlin.util import dataportal

root = os.path.join(os.path.dirname(merlin.__file__), '..', 'test')


def test_read_dax():
    print(root)
    dataPortal = dataportal.LocalDataPortal(
        os.path.join(root, 'auxiliary_files'))
    daxPortal = dataPortal.open_file('test.dax')
    daxReader = imagereader.infer_reader(daxPortal)
    frame0 = daxReader.load_frame(0)
    frame5 = daxReader.load_frame(5)
    frame9 = daxReader.load_frame(9)

    assert daxReader.number_frames == 10
    assert daxReader.image_height == 256
    assert daxReader.image_width == 256
    assert frame0.shape == (256, 256)
    assert frame5.shape == (256, 256)
    assert frame0[0, 0] == 144
    assert frame5[0, 0] == 156
    assert np.sum(frame0) == 10459722
    assert np.sum(frame5) == 10460240
