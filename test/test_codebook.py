import numpy as np
import pytest

from merlin.core import dataset


def test_codebook_get_barcode_count(simple_merfish_data):
    assert simple_merfish_data.get_codebook().get_barcode_count() == 140


def test_codebook_get_bit_count(simple_merfish_data):
    assert simple_merfish_data.get_codebook().get_bit_count() == 16


def test_codebook_get_bit_names(simple_merfish_data):
    for i, n in enumerate(simple_merfish_data.get_codebook().get_bit_names()):
        assert n == 'bit' + str(i+1)


def test_codebook_get_barcode(simple_merfish_data):
    codebook = simple_merfish_data.get_codebook()
    for i in range(codebook.get_barcode_count()):
        assert np.sum(codebook.get_barcode(i)) == 4
    assert np.array_equal(
        codebook.get_barcode(0),
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])


def test_codebook_get_coding_indexes(simple_merfish_data):
    assert np.array_equal(
            simple_merfish_data.get_codebook().get_coding_indexes(), 
            np.arange(70))


def test_codebook_get_blank_indexes(simple_merfish_data):
    assert np.array_equal(
            simple_merfish_data.get_codebook().get_blank_indexes(), 
            np.arange(70, 140))


def test_codebook_get_barcodes(simple_merfish_data):
    bcSetWithBlanks = simple_merfish_data.get_codebook().get_barcodes()
    assert len(bcSetWithBlanks) == 140
    assert all([len(x) == 16 for x in bcSetWithBlanks])
    assert all([np.sum(x) == 4 for x in bcSetWithBlanks])
    bcSetNoBlanks = simple_merfish_data.get_codebook().get_barcodes(
            ignoreBlanks=True)
    assert len(bcSetNoBlanks) == 70
    assert all([len(x) == 16 for x in bcSetNoBlanks])
    assert all([np.sum(x) == 4 for x in bcSetNoBlanks])


def test_codebook_get_name(simple_merfish_data):
    assert simple_merfish_data.get_codebook().get_codebook_name() \
           == 'test_codebook'


def test_codebook_get_index(simple_merfish_data):
    assert simple_merfish_data.get_codebook().get_codebook_index() == 0


def test_codebook_get_gene_names(simple_merfish_data):
    names = simple_merfish_data.get_codebook().get_gene_names()
    codebook = simple_merfish_data.get_codebook()
    for n in names:
        assert n == codebook.get_name_for_barcode_index(
            codebook.get_barcode_index_for_name(n))


def test_two_codebook_save_load(two_codebook_merfish_data):
    codebook1 = two_codebook_merfish_data.get_codebook(0)
    codebook2 = two_codebook_merfish_data.get_codebook(1)
    assert len(two_codebook_merfish_data.get_codebooks()) == 2
    assert codebook1.get_codebook_name() == 'test_codebook2'
    assert codebook1.get_codebook_index() == 0
    assert len(codebook1.get_barcodes()) == 10
    assert codebook2.get_codebook_name() == 'test_codebook'
    assert codebook2.get_codebook_index() == 1
    assert len(codebook2.get_barcodes()) == 140

    reloadedDataset = dataset.MERFISHDataSet(
        'merfish_test', analysisHome='test_analysis_two_codebook')
    reloaded1 = reloadedDataset.get_codebook(0)
    reloaded2 = reloadedDataset.get_codebook(1)
    assert len(reloadedDataset.get_codebooks()) == 2
    assert reloaded1.get_codebook_name() == 'test_codebook2'
    assert reloaded1.get_codebook_index() == 0
    assert len(reloaded1.get_barcodes()) == 10
    assert reloaded2.get_codebook_name() == 'test_codebook'
    assert reloaded2.get_codebook_index() == 1
    assert len(reloaded2.get_barcodes()) == 140

    with pytest.raises(FileExistsError):
        dataset.MERFISHDataSet(
            'merfish_test',
            codebookNames=['test_codebook.csv', 'test_codebook2.csv'],
            analysisHome='test_analysis_two_codebook')
