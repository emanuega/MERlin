import pytest
import shutil
import tempfile
import os
import numpy as np
from botocore import UNSIGNED
from botocore.client import Config
from google.auth.credentials import AnonymousCredentials

from merlin.util import dataportal


def local_data_portal():
    tempPath = tempfile.mkdtemp()
    with open(os.path.join(tempPath, 'test.txt'), 'w') as f:
        f.write('MERlin test file')
    with open(os.path.join(tempPath, 'test.bin'), 'wb') as f:
        f.write(np.array([0, 1, 2], dtype='uint16').tobytes())

    yield dataportal.LocalDataPortal(tempPath)

    shutil.rmtree(tempPath)


def s3_data_portal():
    yield dataportal.S3DataPortal('s3://merlin-test-bucket-vg/test-files',
                                  region_name='us-east-2',
                                  config=Config(signature_version=UNSIGNED))


def gcloud_data_portal():
    yield dataportal.GCloudDataPortal('gc://merlin-test-bucket/test-files',
                                      project='merlin-253419',
                                      credentials=AnonymousCredentials())


@pytest.fixture(scope='function', params=[
    local_data_portal, s3_data_portal, gcloud_data_portal])
def data_portal(request):
    yield next(request.param())


def test_portal_list_files(data_portal):
    # filter out directory blob for google cloud
    fileList = [x for x in data_portal.list_files() if not x.endswith('/')]
    filteredList = data_portal.list_files(extensionList='.txt')
    assert len(fileList) == 2
    assert any([x.endswith('test.txt') for x in fileList])
    assert any([x.endswith('test.bin') for x in fileList])
    assert len(filteredList) == 1
    assert filteredList[0].endswith('test.txt')


def test_portal_available(data_portal):
    assert data_portal.is_available()


def test_portal_read(data_portal):
    textFile = data_portal.open_file('test.txt')
    binFile = data_portal.open_file('test.bin')
    assert textFile.exists()
    assert binFile.exists()
    assert textFile.read_as_text() == 'MERlin test file'
    assert np.array_equal(
        np.frombuffer(binFile.read_file_bytes(0, 6), dtype='uint16'),
        np.array([0, 1, 2], dtype='uint16'))
    assert np.array_equal(
        np.frombuffer(binFile.read_file_bytes(2, 4), dtype='uint16'),
        np.array([1], dtype='uint16'))


def test_exchange_extension(data_portal):
    textFile = data_portal.open_file('test.txt')
    assert textFile.get_file_extension() == '.txt'
    assert textFile.read_as_text() == 'MERlin test file'
    binFile = textFile.get_sibling_with_extension('.bin')
    assert binFile.get_file_extension() == '.bin'
    assert np.array_equal(
        np.frombuffer(binFile.read_file_bytes(0, 6), dtype='uint16'),
        np.array([0, 1, 2], dtype='uint16'))
