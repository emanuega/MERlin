def test_construct_metadataset(simple_metamerfish_data):
    assert simple_metamerfish_data.metaDataSetParameters[
               'overwrite_cached'] == False
    assert ('metamerfish_test_dataset' in
            simple_metamerfish_data.dataSetDict) == 1