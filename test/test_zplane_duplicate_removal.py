import pandas as pd
import random
import numpy as np
from merlin.util import barcodefilters


def generate_barcode(fov, barcode_id, x, y, z, mean_intensity):
    bc = {'barcode': random.getrandbits(32),
          'barcode_id': barcode_id,
          'fov': fov,
          'mean_intensity': mean_intensity,
          'max_intensity': random.uniform(5, 15),
          'area': random.randint(0, 10),
          'mean_distance': random.random(),
          'min_distance': random.random(),
          'x': x,
          'y': y,
          'z': z,
          'global_x': random.uniform(0, 200000),
          'global_y': random.uniform(0, 200000),
          'global_z': random.uniform(0, 5),
          'cell_index': random.randint(0, 5000)}

    for i in range(16):
        bc['intensity_' + str(i)] = random.uniform(5, 15)

    return bc


b1 = generate_barcode(100, 5, 402.21, 787.11, 2, 14.23)
b2 = generate_barcode(100, 5, 502.21, 687.11, 3, 12.23)
b3 = generate_barcode(100, 17, 402.21, 787.11, 2, 10.23)

b1_above_dimmer = generate_barcode(100, 5, 402.21, 787.11, 3, 11.23)
b1_closeby_above_brighter = generate_barcode(100, 5, 403.21, 787.11, 3, 15.23)
b2_above_brighter = generate_barcode(100, 5, 502.31, 687.11, 4, 14.23)
b1_closeby_below_brighter = generate_barcode(100, 5, 403.21, 787.11, 1, 15.0)
b1_closeby_toofar_brighter = generate_barcode(100, 5, 403.21, 787.11, 0, 15.0)


def test_multiple_comparisons_barcodes():
    zplane_cutoff = 1
    xy_cutoff = np.sqrt(2)
    zpositions = [0, 1.5, 3, 4.5, 6, 7.5, 9]

    bcSet = [b1, b2, b3, b1_above_dimmer, b1_closeby_above_brighter,
             b2_above_brighter, b1_closeby_below_brighter,
             b1_closeby_toofar_brighter]
    bcDF = pd.DataFrame(bcSet)
    expected = [x['barcode'] for x in
                [b1_closeby_above_brighter, b2_above_brighter, b3]]
    notExpected = [x['barcode'] for x in [b1, b2, b1_above_dimmer,
                                          b1_closeby_below_brighter,
                                          b1_closeby_toofar_brighter]]

    keptBC = barcodefilters.remove_zplane_duplicates_all_barcodeids(
        bcDF, zplane_cutoff, xy_cutoff, zpositions)
    for ex in expected:
        assert ex in keptBC['barcode'].values
    for notEx in notExpected:
        assert notEx not in keptBC['barcode'].values


def test_all_compatible_barcodes():
    zplane_cutoff = 1
    xy_cutoff = np.sqrt(2)
    zpositions = [0, 1.5, 3, 4.5, 6, 7.5, 9]

    bcSet = [b1, b2, b3, b1_closeby_toofar_brighter]
    bcDF = pd.DataFrame(bcSet)
    expected = [x['barcode'] for x in bcSet]
    keptBC = barcodefilters.remove_zplane_duplicates_all_barcodeids(
        bcDF, zplane_cutoff, xy_cutoff, zpositions)
    for ex in expected:
        assert ex in keptBC['barcode'].values
    assert len(keptBC) == len(bcSet)


def test_farther_zrange():
    zplane_cutoff = 2
    xy_cutoff = np.sqrt(2)
    zpositions = [0, 1.5, 3, 4.5, 6, 7.5, 9]

    bcSet = [b1, b2, b3, b1_closeby_toofar_brighter]
    bcDF = pd.DataFrame(bcSet)
    expected = [x['barcode'] for x in [b2, b3, b1_closeby_toofar_brighter]]
    notExpected = [x['barcode'] for x in [b1]]
    keptBC = barcodefilters.remove_zplane_duplicates_all_barcodeids(
        bcDF, zplane_cutoff, xy_cutoff, zpositions)
    for ex in expected:
        assert ex in keptBC['barcode'].values
    for notEx in notExpected:
        assert notEx not in keptBC['barcode'].values


def test_farther_xyrange():
    zplane_cutoff = 1
    xy_cutoff = np.sqrt(20001)
    zpositions = [0, 1.5, 3, 4.5, 6, 7.5, 9]

    bcSet = [b1, b2, b3]
    bcDF = pd.DataFrame(bcSet)
    expected = [x['barcode'] for x in [b1, b3]]
    notExpected = [x['barcode'] for x in [b2]]
    keptBC = barcodefilters.remove_zplane_duplicates_all_barcodeids(
        bcDF, zplane_cutoff, xy_cutoff, zpositions)
    for ex in expected:
        assert ex in keptBC['barcode'].values
    for notEx in notExpected:
        assert notEx not in keptBC['barcode'].values


def test_empty_barcodes():
    zplane_cutoff = 1
    xy_cutoff = np.sqrt(2)
    zpositions = [0, 1.5, 3, 4.5, 6, 7.5, 9]

    bcDF = pd.DataFrame([b1])
    bcDF.drop(0, inplace=True)

    keptBC = barcodefilters.remove_zplane_duplicates_all_barcodeids(
        bcDF, zplane_cutoff, xy_cutoff, zpositions)
    assert type(keptBC) == pd.DataFrame
