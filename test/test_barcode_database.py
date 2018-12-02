import pytest
from merlin.util import barcodedb

@pytest.fixture(scope='function')
def barcode_db(single_task, simple_merfish_data):
    yield barcodedb.SQLiteBarcodeDB(simple_merfish_data, single_task)

def generate_barcode_dict():
    #return {'barcode': 
    pass

def test_write_and_read_one_fov(barcode_db):
    assert len(barcode_db.get_barcodes()) == 0
