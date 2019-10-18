def test_construct_metadataset(simple_metamerfish_data):
    assert simple_metamerfish_data.metaDataSetParameters[
               'overwrite_cached'] == 0
    assert ('merfish_test' in simple_metamerfish_data.dataSetDict) == 1
