import numpy as np

from merlin.analysis import testtask
from merlin.plots import testplots


def test_metadata(simple_merfish_data):
    randomTask = testtask.RandomNumberParallelAnalysisTask(simple_merfish_data)
    randomMetadata = testplots.TestPlotMetadata(randomTask,
                                                {'test_task': randomTask})
    assert not randomTask.is_complete()
    assert not randomMetadata.is_complete()

    for i in range(randomTask.fragment_count()-1):
        randomTask.run(i)
        randomMetadata.update()
        assert not randomTask.is_complete()
        assert not randomMetadata.is_complete()

    randomTask.run(randomTask.fragment_count()-1)
    randomMetadata.update()
    assert np.isclose(
        randomMetadata.get_mean_values(),
        np.array([np.mean(randomTask.get_random_result(i))
                  for i in range(randomTask.fragment_count())])).all()
    assert randomTask.is_complete()
    assert randomMetadata.is_complete()

