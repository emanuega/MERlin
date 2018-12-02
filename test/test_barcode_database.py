import pytest
from merlin.util import barcodedb

@pytest.fixture(scope='session')
def barcode_db(single_task, simple_data):
    yield barcodedb.SQLiteBarcodeDB(simple_data, single_task)


def test_write_and_read_one_fov(barcode_db):
    assert len(barcode_db.get_barcodes()) == 0
