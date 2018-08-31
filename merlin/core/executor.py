from abc import ABC, abstractmethod
import multiprocessing

from . import analysistask

class Executor(object):

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def run(self):
        pass 


class LocalExecutor(Executor):

    def __init__(self):
        super().__init__()

    def run(self, task, callback=None):
        if isinstance(task, analysistask.ParallelAnalysisTask):
            pool = multiprocessing.Pool(
                    processes=int(multiprocessing.cpu_count()*0.7))
            pool.map_async(
                    task.run, range(task.fragment_count()),
                    callback=callback)

        else:
            pool = multiprocessing.Pool(processes=1)
            pool.apply_async(task.run, callback=callback)
