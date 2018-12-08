import os
import numpy as np

from merlin.data import dataorganization

def test_dataorganization_get_channels(simple_merfish_data):
    assert np.array_equal(
            simple_merfish_data.get_data_organization().get_data_channels(),
            np.arange(18))
            
def test_dataorganization_get_channel_name(simple_merfish_data):
    for i in range(16):
        assert simple_merfish_data.get_data_organization()\
                .get_data_channel_name(i) == 'bit' + str(i+1)

    assert simple_merfish_data.get_data_organization()\
            .get_data_channel_name(16) == 'cellstain'
    assert simple_merfish_data.get_data_organization()\
            .get_data_channel_name(17) == 'nuclearstain'
    
def test_dataorganization_get_fovs(simple_merfish_data):
    assert np.array_equal(
            simple_merfish_data.get_data_organization().get_fovs(),
            np.arange(2))

def test_dataorganization_get_z_positions(simple_merfish_data):
    assert np.array_equal(
            simple_merfish_data.get_data_organization().get_z_positions(),
            np.array([0]))

def test_dataorganization_get_fiducial_information(simple_merfish_data):
    data = simple_merfish_data.get_data_organization()
    for d in data.get_data_channels():
        assert data.get_fiducial_frame_index(d) == 2
    assert os.path.normpath(data.get_fiducial_filename(0, 0)) \
            == os.path.normpath('test_data/merfish_test/test_0_0.tif')
    assert os.path.normpath(data.get_fiducial_filename(0, 1)) \
            == os.path.normpath('test_data/merfish_test/test_1_0.tif')
    assert os.path.normpath(data.get_fiducial_filename(1, 1)) \
            == os.path.normpath('test_data/merfish_test/test_1_0.tif')
    assert os.path.normpath(data.get_fiducial_filename(2, 1)) \
            == os.path.normpath('test_data/merfish_test/test_1_1.tif')

def test_dataorganization_get_image_information(simple_merfish_data):
    data = simple_merfish_data.get_data_organization()
    assert data.get_image_frame_index(0, 0) == 1
    assert data.get_image_frame_index(1, 0) == 0
    assert data.get_image_frame_index(16, 0) == 3
    assert os.path.normpath(data.get_image_filename(0, 0)) \
            == os.path.normpath('test_data/merfish_test/test_0_0.tif')
    assert os.path.normpath(data.get_image_filename(0, 1)) \
            == os.path.normpath('test_data/merfish_test/test_1_0.tif')
    assert os.path.normpath(data.get_image_filename(1, 1)) \
            == os.path.normpath('test_data/merfish_test/test_1_0.tif')
    assert os.path.normpath(data.get_image_filename(2, 1)) \
            == os.path.normpath('test_data/merfish_test/test_1_1.tif')

def test_dataorganization_load_from_dataset(simple_merfish_data):
    originalOrganization = simple_merfish_data.get_data_organization()
    loadedOrganization = dataorganization.DataOrganization(simple_merfish_data)

    assert np.array_equal(originalOrganization.get_data_channels(),
            loadedOrganization.get_data_channels())
    assert np.array_equal(
            originalOrganization.get_fovs(), loadedOrganization.get_fovs())
    assert np.array_equal(originalOrganization.get_z_positions(), 
            loadedOrganization.get_z_positions())

    for channel in originalOrganization.get_data_channels():
        assert originalOrganization.get_data_channel_name(channel) \
                == loadedOrganization.get_data_channel_name(channel)
        assert originalOrganization.get_fiducial_frame_index(channel) \
                == loadedOrganization.get_fiducial_frame_index(channel)

        for fov in originalOrganization.get_fovs():
            assert originalOrganization.get_fiducial_filename(channel, fov) \
                    == loadedOrganization.get_fiducial_filename(channel, fov)
            assert originalOrganization.get_image_filename(channel, fov) \
                    == loadedOrganization.get_image_filename(channel, fov)

        for z in originalOrganization.get_z_positions():
            assert originalOrganization.get_image_frame_index(channel, z) \
                    == loadedOrganization.get_image_frame_index(channel, z)




