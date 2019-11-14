def test_construct_metadataset(simple_metamerfish_data):
    assert ('merfish_test' in simple_metamerfish_data.dataSets) == 1
