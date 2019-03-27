import pytest
import random
import numpy as np
import pandas

from merlin.util import barcodedb


@pytest.fixture(scope='function')
def barcode_db(single_task, simple_merfish_data):
    yield barcodedb.PyTablesBarcodeDB(simple_merfish_data, single_task)


@pytest.fixture(scope='function')
def barcode_db_with_barcodes(barcode_db):
    barcodeSet1 = pandas.DataFrame(
            [generate_random_barcode(0) for i in range(20)])
    barcodeSet2 = pandas.DataFrame(
            [generate_random_barcode(1) for i in range(20)])
    barcodesToWrite = pandas.concat([barcodeSet1, barcodeSet2])
    barcode_db.write_barcodes(barcodesToWrite)

    yield (barcode_db, barcodesToWrite)

    barcode_db.empty_database()
    assert len(barcode_db.get_barcodes()) == 0


def generate_random_barcode(fov):
    randomBarcode = {'barcode': random.getrandbits(32),
        'barcode_id': random.randint(0, 200),
        'fov': fov,
        'mean_intensity': random.uniform(5, 15),
        'max_intensity': random.uniform(5, 15),
        'area': random.randint(0, 10),
        'mean_distance': random.random(),
        'min_distance': random.random(),
        'x': random.uniform(0, 2048),
        'y': random.uniform(0, 2048),
        'z': random.uniform(0, 5),
        'global_x': random.uniform(0, 200000),
        'global_y': random.uniform(0, 200000),
        'global_z': random.uniform(0, 5),
        'cell_index': random.randint(0, 5000)}

    for i in range(16):
        randomBarcode['intensity_' + str(i)] = random.uniform(5, 15)

    return randomBarcode


barcode1 = {'barcode': 290,
        'barcode_id': 0,
        'fov': 0,
        'mean_intensity': 5.0,
        'max_intensity': 7.0,
        'area': 5,
        'mean_distance': 0.1,
        'min_distance': 0.05,
        'x': 10,
        'y': 5,
        'z': 15,
        'global_x': 87,
        'global_y': 29,
        'global_z': 14,
        'cell_index': 8,
        'intensity_0': 89,
        'intensity_1': 89,
        'intensity_2': 89,
        'intensity_3': 89,
        'intensity_4': 89,
        'intensity_5': 89,
        'intensity_6': 89,
        'intensity_7': 89,
        'intensity_8': 89,
        'intensity_9': 89,
        'intensity_10': 89,
        'intensity_11': 89,
        'intensity_12': 89,
        'intensity_13': 89,
        'intensity_14': 89,
        'intensity_15': 54}

barcode2 = {'barcode': 390,
        'barcode_id': 1,
        'fov': 0,
        'mean_intensity': 5.2,
        'max_intensity': 7.2,
        'area': 4,
        'mean_distance': 0.2,
        'min_distance': 0.07,
        'x': 11,
        'y': 6,
        'z': 12,
        'global_x': 81,
        'global_y': 28,
        'global_z': 15,
        'cell_index': 7,
        'intensity_0': 88,
        'intensity_1': 88,
        'intensity_2': 28,
        'intensity_3': 38,
        'intensity_4': 48,
        'intensity_5': 58,
        'intensity_6': 68,
        'intensity_7': 78,
        'intensity_8': 97,
        'intensity_9': 17,
        'intensity_10': 27,
        'intensity_11': 37,
        'intensity_12': 47,
        'intensity_13': 57,
        'intensity_14': 67,
        'intensity_15': 77}


def test_write_and_read_one_fov(barcode_db):
    assert len(barcode_db.get_barcodes()) == 0
    barcodesToWrite = pandas.DataFrame([barcode1, barcode2])
    barcode_db.write_barcodes(barcodesToWrite, fov=0)
    readBarcodes = barcode_db.get_barcodes()
    assert np.isclose(barcodesToWrite.values, readBarcodes.values).all()
    barcode_db.empty_database(fov=0)
    assert len(barcode_db.get_barcodes()) == 0


@pytest.mark.slowtest
def test_write_and_read_one_fov_many_barcodes(barcode_db):
    assert len(barcode_db.get_barcodes()) == 0
    barcodesToWrite = pandas.DataFrame(
            [generate_random_barcode(0) for i in range(200000)])
    barcode_db.write_barcodes(barcodesToWrite, fov=0)
    readBarcodes = barcode_db.get_barcodes()
    assert np.isclose(barcodesToWrite.values, readBarcodes.values).all()
    barcode_db.empty_database(fov=0)
    assert len(barcode_db.get_barcodes()) == 0


def test_multiple_write_one_fov(barcode_db):
    assert len(barcode_db.get_barcodes()) == 0
    barcodeSet1 = pandas.DataFrame(
            [generate_random_barcode(0) for i in range(10)])
    barcodeSet2 = pandas.DataFrame(
            [generate_random_barcode(0) for i in range(10)])
    barcodeSet3 = pandas.DataFrame(
            [generate_random_barcode(0) for i in range(10)])
    barcode_db.write_barcodes(barcodeSet1, fov=0)
    barcode_db.write_barcodes(barcodeSet2, fov=0)
    barcode_db.write_barcodes(barcodeSet3, fov=0)
    readBarcodes = barcode_db.get_barcodes()
    combinedBarcodes = pandas.concat(
            [barcodeSet1, barcodeSet2, barcodeSet3])
    readBarcodes.sort_values(by=list(readBarcodes.columns)[1:], inplace=True)
    combinedBarcodes.sort_values(
            by=list(combinedBarcodes.columns)[1:], inplace=True)
    assert np.isclose(readBarcodes.values, combinedBarcodes.values).all()
    barcode_db.empty_database(fov=0)
    assert len(barcode_db.get_barcodes()) == 0


def test_write_and_read_multiple_fov(barcode_db):
    assert len(barcode_db.get_barcodes()) == 0
    barcodeSet1 = pandas.DataFrame(
            [generate_random_barcode(0) for i in range(10)])
    barcodeSet2 = pandas.DataFrame(
            [generate_random_barcode(1) for i in range(10)])
    combinedBarcodes = pandas.concat([barcodeSet1, barcodeSet2])
    barcode_db.write_barcodes(combinedBarcodes)
    readBarcodes = barcode_db.get_barcodes()
    readBarcodes.sort_values(by=list(readBarcodes.columns)[1:], inplace=True)
    combinedBarcodes.sort_values(
            by=list(combinedBarcodes.columns)[1:], inplace=True)
    assert np.isclose(readBarcodes.values, combinedBarcodes.values).all()
    barcode_db.empty_database()
    assert len(barcode_db.get_barcodes()) == 0


def test_read_select_columns(barcode_db_with_barcodes):
    barcodesInDB = barcode_db_with_barcodes[1]
    readBarcodes = barcode_db_with_barcodes[0].get_barcodes(
            columnList=['mean_intensity', 'x', 'intensity_0'])
    assert np.isclose(
            barcodesInDB[['mean_intensity', 'x', 'intensity_0']].values, 
            readBarcodes.values).all()


def test_read_filtered_barcodes(barcode_db_with_barcodes):
    barcodesInDB = barcode_db_with_barcodes[1]
    for area in range(0, 11, 2):
        for intensity in np.arange(0, 20, 5.1):
            readBarcodes = barcode_db_with_barcodes[0]\
                    .get_filtered_barcodes(area, intensity)
            selectBarcodes = barcodesInDB[
                    (barcodesInDB['area'] >= area) &
                    (barcodesInDB['mean_intensity'] >= intensity)]
            assert len(readBarcodes) == len(selectBarcodes)
            if len(readBarcodes) > 0: 
                readBarcodes.sort_values(
                        by=list(readBarcodes.columns)[1:], inplace=True)
                selectBarcodes = selectBarcodes.sort_values(
                        by=list(selectBarcodes.columns)[1:], inplace=False)
                print(str(area) + ' ' + str(intensity))
                assert np.isclose(
                        readBarcodes.values, selectBarcodes.values).all()


def test_get_barcode_intensities_with_area(barcode_db_with_barcodes):
    barcodesInDB = barcode_db_with_barcodes[1]
    for area in range(11):
        readIntensities = barcode_db_with_barcodes[0]\
                .get_intensities_for_barcodes_with_area(area)
        selectIntensities = barcodesInDB[barcodesInDB['area'] == area]\
                                        ['mean_intensity'].tolist()
        assert np.isclose(
            sorted(readIntensities), sorted(selectIntensities)).all()


@pytest.mark.parametrize('test_function, column_name', [
    ('get_barcode_intensities', 'mean_intensity'),
    ('get_barcode_areas', 'area'),
    ('get_barcode_distances', 'mean_distance'),
])
def test_get_barcode_values(
        barcode_db_with_barcodes, test_function, column_name):
    barcodesInDB = barcode_db_with_barcodes[1]
    readIntensities = getattr(barcode_db_with_barcodes[0], test_function)()
    selectIntensities = barcodesInDB[column_name].tolist()
    assert np.isclose(sorted(readIntensities), sorted(selectIntensities)).all()
